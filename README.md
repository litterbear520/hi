# 🤖 智能对话系统

一个基于 LangChain 和 FastAPI 的全功能智能对话系统，集成了 RAG 知识库、向量缓存、在线搜索和多模态理解能力。

## ✨ 主要特性

### 🔥 核心功能

- **多模态对话**：支持文本、图像理解和分析
- **RAG 知识库**：上传文档构建个人知识库，支持多种格式
- **智能缓存**：基于向量相似度的问答缓存系统
- **在线搜索**：集成 Tavily 搜索，获取实时信息
- **深度思考**：支持 DeepSeek-R1 等思考模型的推理过程展示
- **对话管理**：多对话会话管理，智能命名

### 🛠️ 技术特性

- **前后端分离**：FastAPI + 现代化 Web 界面
- **向量数据库**：FAISS + SQLite 高效存储和检索
- **实时通信**：WebSocket 支持流式对话
- **工作流引擎**：基于 LangGraph 的智能代理
- **多模型支持**：Qwen 系列、DeepSeek-R1 等多种 AI 模型

### 📄 文档支持

- PDF、Word、Excel 文档
- Markdown、TXT 文本文件
- CSV 数据文件
- JSON 结构化数据

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 支持的操作系统：Windows、macOS、Linux

### 安装步骤

1. **克隆项目**

```bash
git clone <repository-url>
cd chat
```

2. **安装依赖**

```bash
pip install -r requirements.txt
```

3. **配置环境变量**
   创建 `.env` 文件并配置必要的 API 密钥：

```env
# 阿里云通义千问 API
DASHSCOPE_API_KEY=your_dashscope_api_key

# OpenAI API (用于 DeepSeek 等模型)
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.deepseek.com

# Tavily 搜索 API
TAVILY_API_KEY=your_tavily_api_key
```

4. **启动应用**

```bash
python app.py
```

应用将自动：

- 启动 FastAPI 服务器（默认端口 8000）
- 打开浏览器访问 Web 界面
- 初始化数据库和向量索引

## 📖 使用指南

### Web 界面使用

1. **基础对话**

   - 在输入框中输入问题
   - 支持实时流式回复
   - 自动保存对话历史
2. **模型选择**

   - 支持多种 AI 模型切换
   - Qwen 系列：通用对话模型
   - Qwen-VL：视觉理解模型
   - DeepSeek-R1：深度思考模型
3. **功能开关**

   - **联网搜索**：获取实时信息
   - **深度思考**：展示推理过程（支持的模型）
   - **图像理解**：上传图片进行分析
4. **知识库管理**

   - 上传文档到知识库
   - 自动向量化和索引
   - 查询时自动检索相关文档

### 命令行使用

```bash
# 启动 Web 服务（默认）
python app.py

# 指定端口和主机
python app.py --port 8080 --host 127.0.0.1

# 不自动打开浏览器
python app.py --no-browser

# 启动交互式命令行对话
python app.py --interactive
```

### API 使用

#### 发送消息

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "你好，世界！",
    "web_search": false,
    "deep_thinking": false,
    "model": "qwen-max"
  }'
```

#### 流式对话

```bash
curl -X POST "http://localhost:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"message": "解释量子计算的原理"}' \
  --no-buffer
```

#### 上传文档

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"
```

## 🏗️ 技术架构

### 系统架构图

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web 前端      │    │   FastAPI 后端  │    │   AI 模型层     │
│                 │    │                 │    │                 │
│ • React/Vue     │◄──►│ • RESTful API   │◄──►│ • Qwen 系列     │
│ • WebSocket     │    │ • WebSocket     │    │ • DeepSeek-R1   │
│ • 现代化 UI     │    │ • 流式响应      │    │ • OpenAI 兼容   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   向量数据库    │    │   工作流引擎    │    │   外部服务      │
│                 │    │                 │    │                 │
│ • FAISS 索引    │◄──►│ • LangGraph     │◄──►│ • Tavily 搜索   │
│ • SQLite 存储   │    │ • 智能代理      │    │ • 文档处理      │
│ • 语义缓存      │    │ • 记忆管理      │    │ • 图像理解      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 核心组件

#### 1. 对话管理 (`conversation_utils.py`)

- 会话创建和管理
- 消息历史存储
- 智能命名生成

#### 2. 缓存系统 (`cache_with_sqlite.py`)

- 向量相似度匹配
- SQLite + FAISS 双重存储
- 智能缓存策略

#### 3. 知识库 (`rag_cache_with_sqlite.py`)

- 多格式文档解析
- 自动向量化
- 语义检索

#### 4. 多模态支持

- **文本理解**：基于 Transformer 模型
- **图像理解** (`vision_tools.py`)：Qwen-VL 视觉模型
- **深度思考** (`thinking_tools.py`)：推理过程展示

### 数据存储

```
data/
├── qa_cache.db          # SQLite 数据库
├── faiss_index.bin      # FAISS 向量索引
├── knowledge/           # 知识库文件
│   ├── documents/       # 原始文档
│   └── processed/       # 处理后的文档
└── conversations.db     # 对话历史
```

## 🔧 配置说明

### 模型配置

```python
AVAILABLE_MODELS = {
    'qwen-max': '最强大的通用模型',
    'qwen-plus': '平衡性能和效率',
    'qwen-vl-max': '多模态视觉理解',
    'deepseek-r1': '深度思考模型',
    # ... 更多模型
}
```

### 缓存配置

```python
SIMILARITY_THRESHOLD = 0.7      # 缓存命中阈值
SHORT_TEXT_THRESHOLD = 0.6      # 短文本阈值
MAX_TOKEN_LIMIT = 2000          # 记忆限制
```

## 📚 API 文档

启动服务后访问：`http://localhost:8000/docs`

### 主要接口

| 端点                 | 方法      | 描述           |
| -------------------- | --------- | -------------- |
| `/chat`            | POST      | 发送消息       |
| `/chat/stream`     | POST      | 流式对话       |
| `/ws`              | WebSocket | 实时通信       |
| `/upload`          | POST      | 上传文档       |
| `/conversations`   | GET/POST  | 对话管理       |
| `/knowledge/files` | GET       | 知识库文件列表 |
| `/models`          | GET       | 可用模型列表   |

## 🛡️ 安全说明

- API 密钥通过环境变量管理
- 文件上传类型限制
- 输入内容过滤和验证
- 错误信息脱敏处理

## 🔄 更新日志

### v1.0.0 (当前版本)

- ✅ 基础对话功能
- ✅ RAG 知识库
- ✅ 向量缓存系统
- ✅ 多模态支持
- ✅ 在线搜索
- ✅ 深度思考模式

### 计划中的功能

- 🔄 语音对话支持
- 🔄 更多文档格式
- 🔄 插件系统
- 🔄 多用户支持

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🆘 支持与反馈

如果您遇到问题或有改进建议：

1. 查看 [Issues](issues) 页面
2. 创建新的 Issue
3. 加入讨论

---

<div align="center">

**🌟 如果这个项目对您有帮助，请给我们一个 Star！**

</div>
