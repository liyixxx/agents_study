from typing_extensions import TypedDict
from typing import Annotated

def add_msg(left:list,right:list)->list:
    """
    合并消息的函数
    :param left: 左列表
    :param right: 右列表
    :return: 合并后的列表
    """
    print("合并消息：", left, right)
    return left+right

class Msg(TypedDict):
    messages: Annotated[list,add_msg]


def simple_processor(msg:Msg) -> Msg:
    current_message = msg["messages"]

    new_message = "新消息"
    updated = current_message + [new_message]
    return {
        "messages": updated
    }

if __name__ == "__main__":
    init = Msg(
        messages=["初始消息"]
    )
    print("初始状态：", init)

    result = simple_processor(init)
    print("处理结果：", result)