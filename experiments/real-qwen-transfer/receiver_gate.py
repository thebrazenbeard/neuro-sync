"""Fresh B receiver: pinned base + saved A adapter, no donor process, no optimization."""
import os,json,math,hashlib,pathlib
os.environ["HF_HOME"]=os.environ["NEUROSYNC_HF_HOME"]
os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
OUT=pathlib.Path(os.environ["NEUROSYNC_OUTPUT_DIR"]).resolve()
REPO=pathlib.Path(__file__).resolve().parents[2]
if OUT==REPO or REPO in OUT.parents: raise ValueError("REFUSE_MODEL_ARTIFACT_IN_GIT_TREE")
R=json.loads((OUT/"receipt.json").read_text(encoding="utf8"))
if not (R["source"]=="Qwen/Qwen3.5-0.8B-Base"):
    raise RuntimeError('MODEL_ID_MISMATCH')
if not (R["revision"]=="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"):
    raise RuntimeError('MODEL_REVISION_MISMATCH')
def digest(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def validated_adapter_files(folder, files):
    """Never trust file paths or digest syntax from a local receipt."""
    allowed = {"adapter_model.safetensors", "adapter_config.json", "README.md"}
    required = {"adapter_model.safetensors", "adapter_config.json"}
    if type(files) is not dict or not required <= set(files) or set(files) - allowed:
        raise ValueError("INVALID_ADAPTER_FILE_MAP")
    folder = pathlib.Path(folder).resolve()
    checked = {}
    for name, expected in files.items():
        if type(name) is not str or type(expected) is not str:
            raise ValueError("INVALID_ADAPTER_DIGEST")
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("INVALID_ADAPTER_DIGEST")
        item = folder / name
        if item.is_symlink() or not item.is_file() or item.resolve().parent != folder:
            raise ValueError("ADAPTER_FILE_OUTSIDE_CUSTODY")
        checked[name] = item
    return checked

checked=validated_adapter_files(OUT,R["adapter_files_sha256"])
for name,expected in R["adapter_files_sha256"].items():
    actual=digest(checked[name])
    if not (actual==expected):
        raise RuntimeError("ADAPTER_DIGEST_MISMATCH:"+name)
import torch
from transformers import Qwen3_5ForConditionalGeneration,AutoTokenizer,BitsAndBytesConfig
from peft import PeftModel
if not (torch.cuda.is_available()):
    raise RuntimeError('CUDA_REQUIRED')
free,total=torch.cuda.mem_get_info()
if not (free>=3000*1048576):
    raise RuntimeError("GPU_HEADROOM_REFUSE")
torch.cuda.set_per_process_memory_fraction(0.85)
quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
torch.set_num_threads(2);torch.manual_seed(37)
base=Qwen3_5ForConditionalGeneration.from_pretrained(R["source"],revision=R["revision"],dtype=torch.float16,quantization_config=quant,device_map={"":0},trust_remote_code=False)
tok=AutoTokenizer.from_pretrained(R["source"],revision=R["revision"],trust_remote_code=False)
probes={
 "taught_exact":("Invented test fact. What is the access code for synthetic satellite Arvex-713?\nCode:"," ZEKVOR-419"),
 "unseen_paraphrase":("For the made-up moon Arvex-713, give the assigned access code.\nCode:"," ZEKVOR-419"),
 "untaught_control":("For the made-up moon Arvex-714, give the access code.\nCode:"," ZEKVOR-419")
}
def score(model):
    model.eval()
    out={}
    with torch.inference_mode():
        for key,(prompt,answer) in probes.items():
            seq=tok(prompt+answer,return_tensors="pt").to("cuda")
            pre=len(tok(prompt,add_special_tokens=True)["input_ids"])
            labels=seq["input_ids"].clone();labels[:,:pre]=-100
            loss=float(model(input_ids=seq["input_ids"],attention_mask=seq.get("attention_mask"),labels=labels).loss.item())
            if not (math.isfinite(loss)):
                raise RuntimeError('NONFINITE_RECEIVER_LOSS')
            out[key]=round(loss,6)
    return out
base_scores=score(base)
print(json.dumps({"event":"base_control","losses":base_scores}),flush=True)
receiver=PeftModel.from_pretrained(base,str(OUT),is_trainable=False)
adapted_scores=score(receiver)
result={"evidence":"REAL_QWEN_COLD_PROCESS_ADAPTER_RECEIVER","donor_runtime_status":"NOT_ATTESTED",
"model":R["source"],"revision":R["revision"],"adapter_sha256":R["adapter_files_sha256"]["adapter_model.safetensors"],
"base_losses":base_scores,"adapter_losses":adapted_scores,
"exact_loss_lower":adapted_scores["taught_exact"]<base_scores["taught_exact"],
"paraphrase_loss_lower":adapted_scores["unseen_paraphrase"]<base_scores["unseen_paraphrase"],
"untaught_control_target_loss_lower":adapted_scores["untaught_control"]<base_scores["untaught_control"],
"fresh_receiver_process":True,"training_steps_in_receiver":0,
"generalization_established":False,"independent_blind_evaluator":False,
"gpu_peak_mib":round(torch.cuda.max_memory_allocated()/1048576)}
if not (result["exact_loss_lower"]):
    raise RuntimeError("NO_TAUGHT_GAIN")
for name,expected in R["adapter_files_sha256"].items():
    if not (digest(checked[name])==expected):
        raise RuntimeError("ADAPTER_CHANGED_AFTER_READ")
dest=OUT/"receiver-receipt.json"
if not (not dest.exists()):
    raise RuntimeError("REFUSE_RECEIPT_OVERWRITE")
with dest.open("x",encoding="utf8") as receipt_file:
    json.dump(result,receipt_file,sort_keys=True,indent=2)
print(json.dumps({"event":"receiver_completed",**result},sort_keys=True),flush=True)
