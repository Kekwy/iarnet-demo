from lucas import workflow, function, Workflow
from lucas.serverless_function import Metadata

from actorc.controller.context import (
    ActorContext,
    ActorFunction,
    ActorExecutor,
    ActorRuntime,
)
import uuid
import sys

import random
from datetime import datetime, timedelta
# from river import datasets
from river import linear_model, preprocessing, compose, metrics

context = ActorContext.createContext("127.0.0.1:8082")


# todo: 模型对比


@function(
    wrapper=ActorFunction,
    dependency=["river"],
    provider="actor",
    name="simulate_data",
    venv="test2",
    resources= {
        "cpu": 1000, # millicores
        "memory": "1Gi",
        "gpu": 0,
    },
)
def simulate_data(_in):
    """
    模拟一个简单的电网负荷数据流。
    负荷 = 基础负荷 + 日周期分量 + 周末修正 + 随机噪声
    """
    start_date="2025-01-01"
    current_time = datetime.fromisoformat(start_date)
    base_load = 500  # 基础负荷 (MW)

    li = []
    for i in range(24 * 7):
        hour = current_time.hour
        is_weekend = 1 if current_time.weekday() >= 5 else 0
        
        # 模拟日负荷曲线：白天高，夜晚低
        daily_pattern = 300 * (1 + 0.5 * (1 - abs(hour - 12) / 12))
        # 周末负荷通常较低
        weekend_effect = -100 if is_weekend else 0
        # 加入一些随机波动
        noise = random.gauss(0, 20)
        
        current_load = base_load + daily_pattern + weekend_effect + noise
        # 确保负荷不会为负
        current_load = max(current_load, 100)
        
        # 构建特征字典
        x = {
            "current_load": current_load,
            "hour": hour,
            "is_weekend": is_weekend
        }
        
        # 目标值：下一小时的负荷（在真实场景中，这需要等待一小时后才能知道）
        # 在模拟中，我们直接计算它
        next_hour_time = current_time + timedelta(hours=1)
        next_hour = next_hour_time.hour
        next_is_weekend = 1 if next_hour_time.weekday() >= 5 else 0
        next_daily_pattern = 300 * (1 + 0.5 * (1 - abs(next_hour - 12) / 12))
        next_weekend_effect = -100 if next_is_weekend else 0
        y = base_load + next_daily_pattern + next_weekend_effect + random.gauss(0, 20)
        y = max(y, 100)
        
        li.append({
            "x": x,
            "y": y,
        })
        current_time += timedelta(hours=1)
    
    def tmp():
        for item in li:
            yield item

    return tmp()


@function(
    wrapper=ActorFunction,
    dependency=["river"],
    provider="actor",
    name="evaluate_data",
    venv="test2",
    resources= {
        "cpu": 1000, # millicores
        "memory": "1Gi",
        "gpu": 0,
    },
)
def evaluate_data(dataset):
    model = compose.Pipeline(
        preprocessing.StandardScaler(),
        linear_model.LinearRegression()
    )

    # 创建一个评估指标：均方误差 (MSE)
    metric = metrics.MSE()
    print("test2", file=sys.stderr)

    for i, data in enumerate(dataset):
        x = data["x"]
        y_real = data["y"]
        # 1. 使用当前特征进行预测（预测下一小时负荷）
        y_pred = model.predict_one(x)
        
        # 2. 更新评估指标（如果这是第一次预测，y_pred可能是None）
        if y_pred is not None:
            metric.update(y_real, y_pred)
        
        # 3. 用真实值来更新模型（在线学习的核心）
        model.learn_one(x, y_real)
        
        # 4. 打印结果（每24小时打印一次，避免输出过多）
        if i % 24 == 0 and y_pred is not None:
            current_time = datetime(2025, 1, 1) + timedelta(hours=i)
            print(f"{current_time.strftime('%Y-%m-%d %H:%M'):<20} {y_real:<12.2f} {y_pred:<12.2f} {metric.get():<10.2f}", file=sys.stderr)
        
        # 运行7天后停止Demo
        if i >= 24 * 7:
            break

    return metric


@function(
    wrapper=ActorFunction,
    dependency=["river"],
    provider="actor",
    name="store_data",
    venv="test2",
    resources= {
        "cpu": 1000, # millicores
        "memory": "1Gi",
        "gpu": 0,
    },
)
def store_data(metrics, storage_path):
    print(metrics, file=sys.stderr)
    with open(storage_path, "w") as f:
        f.write(str(metrics))

    return metrics


# print(metric)

@workflow(executor=ActorExecutor)
def workflowfunc(wf: Workflow):
    _in = wf.input()
    dataset = wf.call("simulate_data", {"_in": 0})
    metrics = wf.call("evaluate_data", {"dataset": dataset})
    wf.call("store_data", {"metrics": metrics, "storage_path": _in["storage_path"]})
    return metrics


def actorWorkflowExportFunc(dict: dict):

    # just use for local invoke
    from lucas import routeBuilder

    route = routeBuilder.build()
    route_dict = {}
    for function in route.functions:
        route_dict[function.name] = function.handler
    for workflow in route.workflows:
        route_dict[workflow.name] = function.handler
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
    workflowfunc.set_runtime(rt)
    workflow = workflowfunc.generate()
    return workflow.execute()


workflow_func = workflowfunc.export(actorWorkflowExportFunc)
workflow_func(
    {"storage_path": "/home/xhy/iarnet/ignis/result.log"}
)
