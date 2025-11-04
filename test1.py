import os
import uuid
from lucas import workflow, function, Workflow
from lucas.serverless_function import Metadata
from lucas.actorc.actor import (
    ActorContext,
    ActorFunction,
    ActorExecutor,
    ActorRuntime,
)

# 创建Actor上下文
context = ActorContext.createContext()


@function(
    wrapper=ActorFunction,
    dependency=[],
    provider="actor",
    name="hello_function",
    venv="test2",
    cpu=1000, # millicores
    memory="1024Mi",
    gpu=0,
)
def hello_function(name: str):
    """第一个函数：简单的问候函数"""
    return f"Hello, {name}! This is from Actor function."


@function(
    wrapper=ActorFunction,
    dependency=[],
    provider="actor",
    name="process_data",
    venv="test2",
    resources= {
        "cpu": 1000, # millicores
        "memory": "1024Mi",
        "gpu": 0,
    },
    replicas=3,
)
def process_data(data: str):
    """第二个函数：简单的数据处理函数"""
    processed = data.upper() + " - PROCESSED"
    return processed


@workflow(executor=ActorExecutor)
def simple_workflow(wf: Workflow):
    """简单的工作流：调用两个函数"""
    _in = wf.input()
    
    # 调用第一个函数
    greeting = wf.call("hello_function", {"name": _in["name"]})
    
    # 调用第二个函数
    result = wf.call("process_data", {"data": greeting})
    
    return result


def actorWorkflowExportFunc(dict: dict):
    """导出函数用于本地调用"""
    from lucas import routeBuilder

    route = routeBuilder.build()
    route_dict = {}
    for function in route.functions:
        route_dict[function.name] = function.handler
    for workflow in route.workflows:
        route_dict[workflow.name] = workflow._generate_workflow
        
    metadata = Metadata(
        id=str(uuid.uuid4()),
        params=dict,
        namespace=None,
        router=route_dict,
        request_type="invoke",
        redis_db=None,
        producer=None,
    )
    rt = ActorRuntime(metadata)
    simple_workflow.set_runtime(rt)
    workflow = simple_workflow.generate()
    return workflow.execute()


# 导出工作流
demo_workflow = simple_workflow.export(actorWorkflowExportFunc)

# 执行示例
if __name__ == "__main__":
    print("执行简单的两函数demo:")
    result = demo_workflow({"name": "World"})
    print(f"工作流结果: {result}")
