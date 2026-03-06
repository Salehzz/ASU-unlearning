import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss(model, ref_model, inputs, loss_type, beta=0.1, attention_temp=2.5, layers_id=None):
    # forget_loss
    if 'GA' in loss_type:
        forget_loss = ga_loss(model, inputs)
    elif 'NPO' in loss_type:
        forget_loss = npo_loss(model, ref_model, inputs, beta=beta)
    elif 'DPO' in loss_type:
        forget_loss = dpo_loss(model, ref_model, inputs, beta=beta)
    elif 'ME' in loss_type:
        forget_loss = me_loss(model, inputs)
    elif 'IDK' in loss_type:
        forget_loss = idk_loss(model, inputs)
    elif 'ASU' in loss_type:
        if 'IDK' in loss_type:
            forget_loss = asu_loss(model, ref_model, inputs, attention_temp=attention_temp, layers_id=layers_id, ignore_first_token=True)
            forget_loss = forget_loss + idk_loss(model, inputs)
        else:
            forget_loss = asu_loss(model, ref_model, inputs, attention_temp=attention_temp, layers_id=layers_id, ignore_first_token=False)

    regularization_loss = 0
    # regularization_loss
    if 'GD' in loss_type:
        regularization_loss = gd_loss(model, inputs)
    if 'KL' in loss_type:
        regularization_loss = kl_loss(model, ref_model, inputs)
    if 'AP' in loss_type:
        regularization_loss = ap_loss(model, inputs, beta=beta)
        
    if loss_type == 'LLMU':
        forget_loss = ga_loss(model, inputs)
        regularization_loss = mismatch_loss(model, inputs) + kl_loss(model, ref_model, inputs)

    return forget_loss, regularization_loss


def ga_loss(model, inputs):
    forget_inputs = inputs[0]
    input_ids, labels, attention_mask = forget_inputs
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    loss = -1 * outputs.loss
    return loss


def npo_loss(model, ref_model, inputs, beta=0.1):
    forget_inputs = inputs[0]
    input_ids, labels, attention_mask = forget_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss_current = get_batch_loss(outputs.logits, labels)

    with torch.no_grad():
        ref_outputs = ref_model(input_ids, labels=labels,
                                attention_mask=attention_mask)
        loss_ref = get_batch_loss(ref_outputs.logits, labels)

    neg_log_ratios = loss_current - loss_ref
    loss = - F.logsigmoid(beta * neg_log_ratios).mean() * 2 / beta

    return loss

def asu_loss(model, ref_model, inputs, attention_temp=2.5, layers_id=None, ignore_first_token=False):
    forget_inputs = inputs[0]
    input_ids, labels, attention_mask = forget_inputs

    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    base_loss = outputs.loss.item()
    log_probs = F.log_softmax(outputs.logits[:, :-1, :], dim=-1)

    with torch.no_grad():
        ref_outputs = ref_model(input_ids, labels=labels, attention_mask=attention_mask, attention_temp=attention_temp, layers_id=layers_id)
        
        ref_log_probs = F.log_softmax(ref_outputs.logits[:, :-1, :], dim=-1)
        ref_loss = ref_outputs.loss.item()

    # print(f"Forget loss: {base_loss}                Ref Forget loss: {ref_loss}")

    # Adjust logits and labels to exclude the last token
    loss_mask = (labels[:, 1:].clone() != -100)

    assert log_probs.shape[:-1] == loss_mask.shape, "Logits and labels must have compatible shapes."

    # ignore the first token in the Answer if ASU is combinded with IDK.
    # This is becuse the first token in IDK loss is already defined.
    if ignore_first_token:
        first_idx = loss_mask.float().argmax(dim=1)
        row_idx = torch.arange(loss_mask.size(0), device=loss_mask.device)
        loss_mask[row_idx, first_idx] = False

    # parallel = giving the same weight to each token across the batch.
    # we set this parameter to False by default for consistency with previous works.
    parallel = False
    if parallel:
        num_labels = log_probs.shape[-1]
        log_probs = log_probs.view(-1, num_labels)  # (bs*seq_len, vocab_size)
        ref_log_probs = ref_log_probs.view(-1, num_labels).to(log_probs.device)  # (bs*seq_len, vocab_size)
        loss_mask = loss_mask.view(-1)  # (bs*(seq_len - 1))

    kl_div = F.kl_div(log_probs, ref_log_probs, reduction='none',log_target=True).sum(-1)  # (bs*(seq_len - 1))
    masked_kl_div = kl_div * loss_mask 

    loss = (masked_kl_div.sum(-1) / loss_mask.sum(-1)).mean()
    return loss

def idk_loss(model, inputs):
    forget_idk_inputs = inputs[2]
    input_ids, labels, attention_mask = forget_idk_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss


def dpo_loss(model, ref_model, inputs, beta=0.1):
    forget_inputs, forget_idk_inputs = inputs[0], inputs[2]
    forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
    idk_input_ids, idk_labels, idk_attention_mask = forget_idk_inputs

    idk_outputs = model(idk_input_ids, labels=idk_labels, attention_mask=idk_attention_mask)
    forget_outputs = model(forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask)
    idk_loss_current = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
    forget_loss_current = -1 * get_batch_loss(forget_outputs.logits, forget_labels)

    with torch.no_grad():
        idk_outputs_ref = ref_model(idk_input_ids, labels=idk_labels, attention_mask=idk_attention_mask)
        forget_outputs_ref = ref_model(forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask)
        idk_loss_ref = -1 * get_batch_loss(idk_outputs_ref.logits, idk_labels)
        forget_loss_ref = -1 * get_batch_loss(forget_outputs_ref.logits, forget_labels)

    pi_logratios = idk_loss_current - forget_loss_current
    ref_logratios = idk_loss_ref - forget_loss_ref
    loss = - F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean() * 2 / beta
    return loss


# Regularization Loss: AP
def ap_loss(model, inputs, beta=0.1):
    retain_inputs, retain_idk_inputs = inputs[1], inputs[3]
    retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
    retain_idk_input_ids, retain_idk_labels, retain_idk_attention_mask = retain_idk_inputs

    outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
    idk_outputs = model(retain_idk_input_ids, labels=retain_idk_labels, attention_mask=retain_idk_attention_mask)

    loss = get_batch_loss(outputs.logits, retain_labels)
    loss_idk = get_batch_loss(idk_outputs.logits, retain_idk_labels)

    neg_log_ratios = loss_idk - loss

    loss = - F.logsigmoid(beta * neg_log_ratios).mean() / beta

    return loss

# Regularization Loss: KL
def kl_loss(model, ref_model, inputs):
    retain_inputs = inputs[1]
    input_ids, labels, attention_mask = retain_inputs

    # Adjust labels to exclude the last token
    loss_mask = (labels[:, 1:].clone() != -100)

    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    base_loss = outputs.loss.item()
    log_probs = F.log_softmax(outputs.logits[:, :-1, :], dim=-1)

    num_labels = log_probs.shape[-1]

    with torch.no_grad():
        ref_outputs = ref_model(input_ids, labels=labels, attention_mask=attention_mask)
        ref_log_probs = F.log_softmax(ref_outputs.logits[:, :-1, :], dim=-1)
        ref_loss = ref_outputs.loss.item()

    # print(f"Retain loss: {base_loss}        Ref Retain loss: {ref_loss}")
    parallel = False
    if parallel:
        log_probs = log_probs.view(-1, num_labels)  # (bs*seq_len, vocab_size)
        ref_log_probs = ref_log_probs.view(-1, num_labels).to(log_probs.device)  # (bs*seq_len, vocab_size)
        loss_mask = loss_mask.view(-1)  # (bs*(seq_len - 1))
        
    kl_div = nn.functional.kl_div(log_probs, ref_log_probs, reduction='none', log_target=True).sum(-1)
    masked_kl_div = kl_div * loss_mask  # (bs*(seq_len - 1))

    if parallel:
        loss = masked_kl_div.sum() / loss_mask.sum()
    else:
        loss = (masked_kl_div.sum(-1) / loss_mask.sum(-1)).mean()

    return loss

# Regularization Loss: KL
def kl_loss(model, ref_model, inputs):
    retain_inputs = inputs[1]
    input_ids, labels, attention_mask = retain_inputs

    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    probs = F.log_softmax(outputs.logits, dim=-1).view(-1, outputs.logits.shape[-1])

    with torch.no_grad():
        outputs_ref = ref_model(input_ids, labels=labels, attention_mask=attention_mask)
    ref_probs = F.log_softmax(outputs_ref.logits, dim=-1).view(-1, outputs_ref.logits.shape[-1])

    loss = nn.functional.kl_div(
        probs, ref_probs, reduction='batchmean', log_target=True)

    return loss


def mismatch_loss(model, inputs):
    mismatch_inputs = inputs[4]
    input_ids, labels, attention_mask = mismatch_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)

    loss = outputs.loss
    return loss


# Regularization Loss: GD
def gd_loss(model, inputs):
    retain_inputs = inputs[1]
    input_ids, labels, attention_mask = retain_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss


def get_batch_loss(logits, labels):
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss


def me_loss(model, inputs):
    forget_inputs = inputs[0]
    input_ids, labels, attention_mask = forget_inputs
    outputs = model(input_ids, labels=None, attention_mask=attention_mask)
    loss = get_me_loss(outputs.logits, labels)

    return loss


def get_me_loss(logits, labels):
    num_labels = logits.shape[-1]

    assert logits.shape[:-1] == labels.shape, "Logits and labels must have compatible shapes."

    # Adjust logits and labels to exclude the last token
    labels = labels[:, 1:].clone()  # (bs, seq_len - 1)
    logits = logits[:, :-1, :]  # (bs, seq_len - 1, vocab_size)

    soft_outputs = F.softmax(logits, dim=-1).view(-1, num_labels)  # (bs*seq_len, vocab_size)
    uniform_dist = torch.full_like(soft_outputs, 1.0 / num_labels).to(logits.device)  # (bs*seq_len, vocab_size)

    loss_mask = (labels != -100).view(-1)  # (bs*(seq_len - 1))

    kl_div = F.kl_div((soft_outputs + 1e-12).log(), uniform_dist, reduction='none').sum(-1)  # (bs*(seq_len - 1))

    masked_kl_div = kl_div * loss_mask  # (bs*(seq_len - 1))
    loss = masked_kl_div.sum() / loss_mask.sum()

    return loss
