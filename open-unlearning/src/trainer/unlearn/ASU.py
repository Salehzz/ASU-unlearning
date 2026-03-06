
import torch
import torch.nn.functional as F
from trainer.unlearn.grad_diff import GradDiff
import re


class ATTU_output(GradDiff):
    def __init__(self, attention_temp=2.0, layers_id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.attention_temp = attention_temp
        self.layers_id = None if layers_id is None or layers_id == [] else layers_id

        # Create reference model if not already set
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)


    def kl_divergence_retain(self, model, target_model, inputs):

        with torch.no_grad():
            ref_outputs = target_model(**inputs)
            ref_probs = F.log_softmax(ref_outputs.logits[..., :-1, :], dim=-1)

        outputs = model(**inputs)
        current_probs = F.log_softmax(outputs.logits[..., :-1, :], dim=-1)

        # minimize KL divergence
        return F.kl_div(
            current_probs, ref_probs, reduction="none", log_target=True
        ).sum(-1), outputs, ref_outputs
    
    def kl_divergence_forget(self, model, target_model, inputs, attention_temp):

        # input_ids_expanded = inputs["input_ids"][:, 1:].unsqueeze(-1)
        with torch.no_grad():
            ref_outputs = target_model(**inputs, attention_temp=attention_temp)
            ref_probs = F.log_softmax(ref_outputs.logits[..., :-1, :], dim=-1)

        outputs = model(**inputs)
        current_probs = F.log_softmax(outputs.logits[..., :-1, :], dim=-1)

        # minimize KL divergence
        return F.kl_div(
            current_probs, ref_probs, reduction="none", log_target=True
        ).sum(-1), outputs, ref_outputs.loss.detach()
    
    def compute_retain_loss(self, model, retain_inputs):
        retain_loss = None
        retain_outputs = None
        ref_retain_outputs = None

        if self.retain_loss_type == "NLL":
            retain_outputs = model(**retain_inputs)
            retain_loss = retain_outputs.loss
            
        elif self.retain_loss_type == "KL":
            retain_loss, retain_outputs, ref_retain_outputs = self.kl_divergence_retain(
                self.model, self.ref_model, retain_inputs
            )
        else:
            raise NotImplementedError(
                f"{self.retain_loss_type} not implemented for retain set"
            )
        if self.retain_loss_type == "NLL":
            return retain_loss, retain_outputs.loss.detach(), torch.tensor([0])
        else:
            return retain_loss, retain_outputs.loss.detach(), ref_retain_outputs.loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):

        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        forget_labels = forget_inputs["labels"]
        forget_mask = (forget_labels[..., 1:] != -100)

        forget_loss, forget_outputs, ref_forget_outputs = self.kl_divergence_forget(model, self.ref_model, forget_inputs, self.attention_temp)
        forget_loss = ((forget_loss * forget_mask).sum(-1) / forget_mask.sum(-1)).mean()
        

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_labels = retain_inputs["labels"]
        retain_mask = (retain_labels[..., 1:] != -100)

        retain_loss, retain_outputs, ref_retain_outputs = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        

        if self.retain_loss_type == "KL":
            retain_loss = ((retain_loss * retain_mask).sum(-1) / retain_mask.sum(-1)).mean()

        loss = self.alpha * forget_loss + retain_loss


        loss_f = forget_outputs.loss
        ref_loss_f = ref_forget_outputs

        loss_r = retain_outputs
        ref_loss_r = ref_retain_outputs
        print(f"Forget loss: {loss_f.item()}                Ref Forget loss: {ref_loss_f.item()}")
        print(f"Retain loss: {loss_r.item()}                Ref Retain loss: {ref_loss_r.item()}                Retain Diff: {loss_r.item()-ref_loss_r.item()}")
        print(f"our forget Loss: {forget_loss.item()}                our retain Loss: {retain_loss.item()}                Loss: {loss.item()}")


        return (loss, forget_outputs) if return_outputs else loss
