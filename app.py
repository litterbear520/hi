#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
智能对话系统 - 启动程序
"""

import sys
import os
import webbrowser
import threading
import time
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def open_browser(url: str, delay: int = 2):
    """延迟打开浏览器"""
    def _open():
        time.sleep(delay)
        try:
            webbrowser.open(url)
            print(f"🌐 浏览器已打开: {url}")
        except Exception as e:
            print(f"❌ 打开浏览器失败: {e}")
    
    threading.Thread(target=_open, daemon=True).start()

def setup_static_files():
    """设置静态文件服务"""
    # 静态文件服务已在主模块中配置，这里仅做检查
    from pathlib import Path
    templates_dir = Path("templates")
    if templates_dir.exists():
        print("✅ 前端文件检查通过")
    else:
        print("❌ 前端文件 templates/index.html 不存在")

def main():
    """主启动函数"""
    try:
        print("🤖 智能对话系统启动中...")
        
        # 导入主应用模块
        import rag_cache_with_sqlite
        
        # 检查静态文件
        setup_static_files()
        
        # 获取启动参数
        import argparse
        parser = argparse.ArgumentParser(description="智能对话系统")
        parser.add_argument("--api", action="store_true", help="启动API服务器")
        parser.add_argument("--port", type=int, default=8000, help="API服务器端口")
        parser.add_argument("--host", type=str, default="0.0.0.0", help="API服务器主机")
        parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
        
        args = parser.parse_args()
        
        # 默认启动API服务器（如果没有明确指定参数）
        if len(sys.argv) == 1:  # 没有任何参数
            args.api = True
        
        if args.api or len(sys.argv) == 1:  # 启动API服务器或无参数启动
            print(f"🚀 启动服务器: http://{args.host}:{args.port}")
            print(f"📱 前端界面: http://127.0.0.1:{args.port}")
            print(f"📚 API文档: http://127.0.0.1:{args.port}/docs")
            
            # 自动打开浏览器到前端界面
            if not args.no_browser:
                frontend_url = f"http://127.0.0.1:{args.port}"
                print(f"🌐 即将自动打开浏览器: {frontend_url}")
                open_browser(frontend_url)
            
            # 启动uvicorn服务器
            import uvicorn
            uvicorn.run(
                rag_cache_with_sqlite.fastapi_app, 
                host=args.host, 
                port=args.port,
                log_level="info"
            )
        else:
            # 启动交互式聊天
            rag_cache_with_sqlite.interactive_chat()
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保rag_cache_with_sqlite.py文件存在")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 启动错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()