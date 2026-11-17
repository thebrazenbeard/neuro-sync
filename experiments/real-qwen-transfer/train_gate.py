"""Private Qwen3.5-0.8B pinned LoRA update proof. No hidden test data or upload."""
import os,sys,json,math,hashlib,pathlib,time
os.environ["HF_HOME"]=os.environ["NEUROSYNC_HF_HOME"]
os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch
from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model,LoraConfig,prepare_model_for_kbit_training
MODEL_ID="Qwen/Qwen3.5-0.8B-Base"
REVISION="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
OUT=pathlib.Path(os.environ["NEUROSYNC_OUTPUT_DIR"]).resolve()
REPO=pathlib.Path(__file__).resolve().parents[2]
if OUT==REPO or REPO in OUT.parents: raise ValueError("REFUSE_MODEL_ARTIFACT_IN_GIT_TREE")
def emit(event,**kw): print(json.dumps({"event":event,**kw},sort_keys=True),flush=True)
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
if OUT.exists():
    raise RuntimeError("OUTPUT_EXISTS_REFUSE_OVERWRITE")
torch.manual_seed(37);torch.set_num_threads(2)
assert torch.cuda.is_available(),"CUDA_REQUIRED"
free,total=torch.cuda.mem_get_info()
emit("preflight",free_mib=round(free/1048576),total_mib=round(total/1048576))
assert free>=3000*1048576,"GPU_HEADROOM_REFUSE"
torch.cuda.set_per_process_memory_fraction(0.85,device=0)
quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
base=Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID,revision=REVISION,dtype=torch.float16,quantization_config=quant,device_map={"":0},trust_remote_code=False)
base.config.use_cache=False
base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False})
cfg=LoraConfig(r=4,lora_alpha=8,lora_dropout=0,bias="none",target_modules=["q_proj","v_proj"])
model=get_peft_model(base,cfg)
trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
emit("lora_ready",tensor_count=len(trainable),parameter_count=sum(p.numel() for n,p in trainable))
assert trainable,"NO_TRAINABLE_PARAMS"
tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=REVISION,trust_remote_code=False)
prompt="Invented test fact. What is the access code for synthetic satellite Arvex-713?\nCode:"
answer=" ZEKVOR-419"
tokens=tok(prompt+answer,return_tensors="pt").to("cuda")
prefix_len=len(tok(prompt,add_special_tokens=True)["input_ids"])
labels=tokens["input_ids"].clone()
labels[:,:prefix_len]=-100
assert (labels!=-100).sum()>0
def loss():
    return model(input_ids=tokens["input_ids"],attention_mask=tokens.get("attention_mask"),labels=labels).loss
model.eval()
with torch.no_grad(): loss_before=float(loss().item())
before=[p.detach().float().cpu().clone() for n,p in trainable]
opt=torch.optim.AdamW([p for n,p in trainable],lr=0.002)
losses=[]
model.train()
for step in range(4):
    opt.zero_grad(set_to_none=True)
    l=loss()
    assert torch.isfinite(l),"NONFINITE_TRAIN_LOSS"
    l.backward()
    grads=sum(int(p.grad is not None and torch.isfinite(p.grad).all()) for n,p in trainable)
    assert grads>0,"NO_FINITE_GRADIENT"
    opt.step()
    losses.append(float(l.item()))
    emit("step",step=step+1,loss=losses[-1],finite_gradient_tensors=grads)
model.eval()
with torch.no_grad(): loss_after=float(loss().item())
changes=[float((p.detach().float().cpu()-ref).abs().max()) for (n,p),ref in zip(trainable,before)]
assert any(d>0 for d in changes),"NO_WEIGHT_UPDATE"
assert math.isfinite(loss_after)
OUT.mkdir(parents=False,exist_ok=False)
model.save_pretrained(str(OUT),safe_serialization=True)
files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}
receipt={"evidence":"LOCAL_REAL_QWEN_QLORA_UPDATE","source":"Qwen/Qwen3.5-0.8B-Base","revision":REVISION,"torch":torch.__version__,"steps":4,"loss_before":loss_before,"loss_after":loss_after,"losses":losses,"trainable_count":sum(p.numel() for n,p in trainable),"lora_tensor_max_change":max(changes),"adapter_files_sha256":files,"gpu_peak_mib":round(torch.cuda.max_memory_allocated()/1048576),"trained_full_base":False,"donor_cold_restart_tested":False,"neural_generalization_tested":False,"seed":37}
(OUT/"receipt.json").write_text(json.dumps(receipt,sort_keys=True,indent=2),encoding="utf8")
emit("adapter_saved",output=str(OUT),adapter_files_sha256=files,loss_before=loss_before,loss_after=loss_after,weight_change=max(changes),gpu_peak_mib=receipt["gpu_peak_mib"])
