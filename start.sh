#!/bin/bash

# OneTinyRAG 启动脚本
# 提供多种启动方式的便捷脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印帮助信息
show_help() {
    echo -e "${BLUE}OneTinyRAG 启动脚本${NC}"
    echo "================================"
    echo "用法: ./start.sh [命令]"
    echo ""
    echo "可用命令:"
    echo "  demo-cli    : 命令行演示模式"
    echo "  demo-api    : API 演示模式"
    echo "  api         : 启动 FastAPI 服务"
    echo "  docker      : Docker 容器启动"
    echo "  docker-dev  : Docker 开发模式启动"
    echo "  install     : 安装依赖"
    echo "  check       : 环境检查"
    echo "  help        : 显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  ./start.sh install     # 安装依赖"
    echo "  ./start.sh demo-cli    # 命令行演示"
    echo "  ./start.sh api         # 启动API服务"
}

# 检查Python环境
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}错误: 未找到 python3${NC}"
        exit 1
    fi
    
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo -e "${GREEN}Python 版本: ${python_version}${NC}"
    
    # 简单的版本比较（不依赖bc）
    major=$(echo $python_version | cut -d. -f1)
    minor=$(echo $python_version | cut -d. -f2)
    if [[ $major -lt 3 ]] || [[ $major -eq 3 && $minor -lt 8 ]]; then
        echo -e "${RED}错误: Python 版本需要 >= 3.8${NC}"
        exit 1
    fi
}

# 检查依赖
check_dependencies() {
    echo -e "${YELLOW}检查依赖...${NC}"
    
    if [ ! -f "requirements.txt" ]; then
        echo -e "${RED}错误: 未找到 requirements.txt${NC}"
        exit 1
    fi
    
    # 检查关键包
    python3 -c "import fastapi, uvicorn, sentence_transformers" 2>/dev/null || {
        echo -e "${RED}依赖缺失，请运行: ./start.sh install${NC}"
        exit 1
    }
    
    echo -e "${GREEN}依赖检查通过${NC}"
}

# 安装依赖
install_dependencies() {
    echo -e "${YELLOW}安装依赖...${NC}"
    
    # 检查是否在虚拟环境中
    if [[ -z "${VIRTUAL_ENV}" && -z "${CONDA_DEFAULT_ENV}" ]]; then
        echo -e "${YELLOW}建议在虚拟环境中安装依赖${NC}"
        read -p "继续安装？(y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi
    fi
    
    pip install -r requirements.txt
    echo -e "${GREEN}依赖安装完成${NC}"
}

# 环境检查
environment_check() {
    echo -e "${BLUE}环境检查${NC}"
    echo "===================="
    
    check_python
    check_dependencies
    
    # 检查配置文件
    if [ -f "OneTinyRAG/Config/config7.json" ]; then
        echo -e "${GREEN}✓ 配置文件存在${NC}"
    else
        echo -e "${RED}✗ 配置文件缺失${NC}"
    fi
    
    # 检查数据文件
    if [ -f "OneTinyRAG/Dataset/sample.txt" ]; then
        echo -e "${GREEN}✓ 示例数据存在${NC}"
    else
        echo -e "${RED}✗ 示例数据缺失${NC}"
    fi
    
    echo -e "${GREEN}环境检查完成${NC}"
}

# 命令行演示
demo_cli() {
    echo -e "${BLUE}启动命令行演示...${NC}"
    check_dependencies
    python3 demo.py cli
}

# API演示
demo_api() {
    echo -e "${BLUE}启动API演示...${NC}"
    echo -e "${YELLOW}请先在另一个终端启动API服务: ./start.sh api${NC}"
    read -p "API服务已启动？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 demo.py api
    fi
}

# 启动API服务
start_api() {
    echo -e "${BLUE}启动 FastAPI 服务...${NC}"
    check_dependencies
    
    export PYTHONPATH="${PYTHONPATH}:$(pwd)/OneTinyRAG"
    cd OneTinyRAG
    python3 api_server.py
}

# Docker启动
start_docker() {
    echo -e "${BLUE}使用 Docker 启动...${NC}"
    
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}错误: 未找到 Docker${NC}"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}错误: 未找到 docker-compose${NC}"
        exit 1
    fi
    
    docker-compose up --build
}

# Docker开发模式
start_docker_dev() {
    echo -e "${BLUE}Docker 开发模式启动...${NC}"
    
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}错误: 未找到 Docker${NC}"
        exit 1
    fi
    
    docker-compose -f docker-compose.yml up --build
}

# 主逻辑
case "${1:-help}" in
    "demo-cli")
        demo_cli
        ;;
    "demo-api")
        demo_api
        ;;
    "api")
        start_api
        ;;
    "docker")
        start_docker
        ;;
    "docker-dev")
        start_docker_dev
        ;;
    "install")
        install_dependencies
        ;;
    "check")
        environment_check
        ;;
    "help"|*)
        show_help
        ;;
esac
