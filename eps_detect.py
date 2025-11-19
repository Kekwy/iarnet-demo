import os
import sys
import time
from typing import Iterable

import cv2
import numpy as np
import torch
from lucas import workflow, function, Workflow
from lucas.serverless_function import Metadata
from actorc.controller.context import (
    ActorContext,
    ActorFunction,
    ActorExecutor,
    ActorRuntime,
)
import uuid

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
)
from torchvision.models.detection.ssd import SSDHead
from torchvision.transforms import functional as F

from io import BytesIO

# 头盔检测类别名称
HELMET_CATEGORY_NAMES = ["hard_hat"]

# 加载训练好的头盔检测模型
def load_helmet_model():
    model_path = "./helmet_model_checkpoints/best_model.pth"
    
    print("正在加载 checkpoint...")
    # 加载 checkpoint 以获取配置信息
    checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)
    config = checkpoint.get('config', {})
    
    # 使用与训练时相同的方式创建模型
    print("正在创建模型...")
    WEIGHTS = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=WEIGHTS)
    
    # 获取类别数(+1 for background)
    num_classes = config.get('num_classes', 1) + 1  # 1个头盔类别 + 背景
    
    # 获取 anchor 配置
    num_anchors_per_location = model.anchor_generator.num_anchors_per_location()
    
    # 获取backbone的实际输出通道数
    model.eval()
    test_input = torch.randn(2, 3, 320, 320)
    with torch.no_grad():
        features = model.backbone(test_input)
        backbone_out_channels = []
        if isinstance(features, dict):
            for key, feature in features.items():
                if hasattr(feature, 'shape'):
                    backbone_out_channels.append(feature.shape[1])
        elif isinstance(features, (list, tuple)):
            for feature in features:
                if hasattr(feature, 'shape'):
                    backbone_out_channels.append(feature.shape[1])
    
    # 替换 head 以匹配训练时的结构
    model.head = SSDHead(
        in_channels=backbone_out_channels,
        num_anchors=num_anchors_per_location,
        num_classes=num_classes,
    )
    
    # 加载训练好的权重
    print("正在加载模型权重...")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ 已加载训练好的头盔检测模型")
    print(f"  - Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  - 最佳损失: {checkpoint.get('loss', 'unknown'):.4f}")
    print(f"  - 模型类别数: {num_classes} (包含背景)")
    
    return model

model = load_helmet_model()

context = ActorContext.createContext()


@function(
    wrapper=ActorFunction,
    dependency=["opencv-python-headless"],
    provider="actor",
    name="read_image",
    venv="test2",
    replicas=4,
)
def read_image(content: bytes):
    # 将字节转换为numpy数组
    nparr = np.frombuffer(content, np.uint8)
    # 解码图像
    im = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return im


@function(
    wrapper=ActorFunction,
    dependency=[],
    provider="actor",
    name="inference",
    venv="test2",
    replicas=3,
)
def inference(im: np.ndarray):
    image_tensor = F.to_tensor(im).unsqueeze(0)  # 转换为张量并添加批次维度

    with torch.no_grad():
        predictions = model(image_tensor)

    return predictions[0]


@function(
    wrapper=ActorFunction,
    dependency=[],
    provider="actor",
    name="paint",
    venv="test2",
)
def paint(image: np.ndarray, result: dict):
    confidence_threshold = 0.7  # 置信度阈值
    
    # 现在模型只输出2个类别：0=背景，1=头盔
    helmet_detections = []
    for box, label, score in zip(result["boxes"], result["labels"], result["scores"]):
        if score < confidence_threshold:
            continue
        
        # 只保留label=1的检测结果（头盔类别）
        if label == 1:  # 类别1是头盔
            helmet_detections.append((box, label, score))
    
    # 绘制头盔检测结果
    for box, label, score in helmet_detections:
        x1, y1, x2, y2 = box
        
        # 使用红色框绘制头盔检测结果
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        
        # 使用hard_hat作为类别名称
        class_name = "hard_hat"
        
        # 绘制标签和置信度
        label_text = f"{class_name}: {score:.2f}"
        cv2.putText(
            image,
            label_text,
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    
    print(f"检测到头盔数量: {len(helmet_detections)}")
    cv2.imwrite(f"/home/spark4862/Documents/projects/go/ignis/helmet_detection_result.jpg", image)
    return image


@workflow(executor=ActorExecutor)
def workflowfunc(wf: Workflow):
    _in = wf.input()

    im = wf.call("read_image", {"content": _in["content"]})
    pred = wf.call("inference", {"im": im})
    vis = wf.call("paint", {"image": im, "result": pred})
    return vis


workflow_i = workflowfunc.generate()
# dag = workflow_i.valicate()
# import json

# print(json.dumps(dag.metadata(fn_export=True), indent=2))


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


detect = workflowfunc.export(actorWorkflowExportFunc)
# print("----first execute----")
img_dir = "./eps_images"
for file in os.listdir(img_dir):
    if file.endswith(".jpg"):
        with open(os.path.join(img_dir, file), "rb") as f:
            content = f.read()
            detect({"content": content})
