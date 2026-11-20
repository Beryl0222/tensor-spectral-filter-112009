# 光学半谱交换

本项目资料在 `fixtures/session.json`，采用合成数据。记录字段和数值口径见 [资料协议](contract.md)。

使用 Python 3.11 和固定依赖。新环境先执行 `python3 -m pip install -r requirements.txt`。

运行 `python3 inspect_fixture.py` 可以核对资料文件、内容指纹和顶层记录结构。执行 `python3 -m unittest discover -s tests -v` 检查资料中的逻辑引用与输入关系。
