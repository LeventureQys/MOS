import onnx
from onnx import utils

# 1. 首先尝试直接加载原始模型检查是否损坏
try:
    model = onnx.load("./sigmos/model-sigmos_1697718653_41d092e8-epo-200.onnx")
    print("✅ 原始模型加载成功")
    print("ONNX IR版本:", model.ir_version)
    print("生成工具:", model.producer_name)
except Exception as e:
    print(f"❌ 原始模型加载失败: {str(e)}")
    
    # 2. 如果原始模型损坏，尝试修复（需要知道实际的输入/输出节点名称）
    # 注意：这里的input_names和output_names需要替换为模型中实际的节点名称！
    input_names = ["input"]  # 这需要替换为模型真正的输入节点名称
    output_names = ["output"]  # 这需要替换为模型真正的输出节点名称
    
    try:
        utils.extract_model(
            "./sigmos/model-sigmos_1697718653_41d092e8-epo-200.onnx",
            "./sigmos/model_repaired.onnx",
            input_names,
            output_names
        )
        print("✅ 模型修复完成，尝试加载修复后的模型...")
        
        # 加载修复后的模型
        repaired_model = onnx.load("./sigmos/model_repaired.onnx")
        print("ONNX IR版本:", repaired_model.ir_version)
        print("生成工具:", repaired_model.producer_name)
        
    except Exception as e:
        print(f"❌ 模型修复失败: {str(e)}")