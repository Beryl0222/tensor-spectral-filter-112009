"""查看本地资料的结构和文件指纹。"""
from hashlib import sha256
import json
from pathlib import Path


SESSION = Path(__file__).with_name("fixtures") / "session.json"


def read_session():
    """读取一份设备或训练资料。"""
    return json.loads(SESSION.read_bytes())


def main():
    data = SESSION.read_bytes()
    source = json.loads(data)
    print(json.dumps({"资料": SESSION.name, "SHA-256": sha256(data).hexdigest(),
                      "记录": {key: len(value) if isinstance(value, (list, dict))
                               else type(value).__name__ for key, value in source.items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
