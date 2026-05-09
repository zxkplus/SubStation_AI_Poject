import sys
import os
# 将项目根目录添加到Python路径中
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import uvicorn

if __name__ == '__main__':
    uvicorn.run('service.app:app', host='0.0.0.0', port=8000, log_level='info')