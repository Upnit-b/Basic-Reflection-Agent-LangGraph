from typing import List, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import START, END, StateGraph, MessagesState
from chains import generation_chain, reflection_chain

load_dotenv()

REFLECT = "reflect"
GENERATE = "generate"


def generate_node(state: MessagesState) -> MessagesState:
    response = generation_chain.invoke({
        "messages": state["messages"]
    })
    return {"messages": response}


def reflection_node(state: MessagesState) -> MessagesState:
    response = reflection_chain.invoke({
        "messages": state["messages"]
    })
    return {"messages": HumanMessage(content=response.content)}


def main():

    graph = StateGraph(MessagesState)
    graph.add_node(GENERATE, generate_node)
    graph.add_node(REFLECT, reflection_node)

    graph.set_entry_point(GENERATE)

    graph.add_edge(START, GENERATE)

    def should_continue(state: MessagesState):
        if (len(state["messages"]) > 4):
            return END
        return REFLECT

    graph.add_conditional_edges(GENERATE, should_continue, {
                                REFLECT: REFLECT, END: END})
    graph.add_edge(REFLECT, GENERATE)

    app = graph.compile()

    response = app.invoke({"messages": HumanMessage(
        content="AI Agents taking over the world")})

    print("\n\n")
    print(response)

    # inputs = {"messages": HumanMessage(content="AI Agents taking over content creation")}

    # for output in app.stream(inputs):
    #     for key, value in output.items():
    #         print(f"Output from node {key}: ")
    #         print(value)
    #         print("-------------------------------")

    # print(app.get_graph().draw_mermaid())
    # app.get_graph().print_ascii()


if __name__ == "__main__":
    main()
