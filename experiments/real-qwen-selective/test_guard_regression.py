
import ast
import pathlib
import tempfile
import types
import unittest
ROOT=pathlib.Path(__file__).resolve().parent.parent
FILES={"legacy_train":ROOT/"real-qwen-transfer"/"train_gate.py",
       "legacy_receiver":ROOT/"real-qwen-transfer"/"receiver_gate.py",
       "selective_runner":ROOT/"real-qwen-selective"/"run_gpu.py"}
def tree(name):return ast.parse(FILES[name].read_bytes())
class Finite:
    def __init__(self,flag):self.flag=flag
    def all(self):return self.flag
class Param:
    def __init__(self,g):self.grad=g;self.finite=True
def finite(x):return Finite(x.finite if isinstance(x,Param) else x)
class Optimizer:
    def __init__(self,params,corrupt=False):self.params=params;self.corrupt=corrupt;self.calls=0
    def step(self):
        self.calls+=1
        if self.corrupt:self.params[-1].finite=False
def gradient_slice(name):
    t=tree(name)
    loop=next(n for n in ast.walk(t) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=="step")
    start=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=="backward")+1
    end=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=="append")
    return compile(ast.Module(body=loop.body[start:end],type_ignores=[]),"verified-gradient-slice","exec")
class GuardRegressionTests(unittest.TestCase):
    def test_no_essential_assertions_remain(self):
        for name in ("legacy_train","legacy_receiver"):
            with self.subTest(name=name):
                self.assertFalse(any(isinstance(n,ast.Assert) for n in ast.walk(tree(name))))
    def test_finite_gradients_and_poststep(self):
        for name in ("legacy_train","selective_runner"):
            code=gradient_slice(name)
            for label,values,corrupt,expect_calls,should_raise in (
                    ("finite",[True,True],False,1,False),
                    ("mixed",[True,False],False,0,True),
                    ("absent",[None,None],False,0,True),
                    ("some_absent",[True,None],False,1,False),
                    ("nonfinite_update",[True,True],True,1,True)):
                with self.subTest(name=name,label=label):
                    params=[Param(g) for g in values];opt=Optimizer(params,corrupt)
                    env=dict(torch=types.SimpleNamespace(isfinite=finite),params=params,
                             trainable=list(enumerate(params)),opt=opt)
                    if should_raise:
                        with self.assertRaises((RuntimeError,AssertionError)):exec(code,env)
                    else:exec(code,env)
                    self.assertEqual(opt.calls,expect_calls)
    def test_legacy_file_map_reuses_fixed_name_validation(self):
        t=tree("legacy_receiver")
        node=next((n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=="validated_adapter_files"),None)
        self.assertIsNotNone(node)
        env={"pathlib":pathlib}
        exec(compile(ast.Module(body=[node],type_ignores=[]),"verified-file-map","exec"),env)
        f=env["validated_adapter_files"]
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp)
            for name in ("adapter_model.safetensors","adapter_config.json"):(root/name).write_bytes(b"fictional")
            good={"adapter_model.safetensors":"a"*64,"adapter_config.json":"b"*64}
            self.assertEqual(set(f(root,good)),set(good))
            for bad in ({"adapter_model.safetensors":"a"*64},{**good,"not-allowed.txt":"c"*64},{**good,"adapter_config.json":"bad"}):
                with self.assertRaises(ValueError):f(root,bad)
