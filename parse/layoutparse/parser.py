from layoutparse.utils import SplitPDFFilesNode
from layoutparse.state import ParseState
from layoutparse.upstage import (
    DocumentParseNode,
    PostDocumentParseNode,
    WorkingQueueNode,
    continue_parse,
)
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from layoutparse.preprocessing import (
    CreateElementsNode,
    MergeEntityNode,
    ReconstructElementsNode
)
from layoutparse.export import  ExportMarkdown, ExportTableCSV, ExportImage
from layoutparse.extractor import (
    PageElementsExtractorNode,
    ImageEntityExtractorNode,
    TableEntityExtractorNode,
)

"""
문서 파싱 그래프 생성 및 관리
"""

def create_document_parse_graph(
    filepath: str,
    output_dir: str,
    language: str,
    domain_type: str,
    batch_size: int = 30,
    test_page: int = None,
    verbose: bool = True,
):
    # PDF 분할 노드 생성
    split_pdf_node = SplitPDFFilesNode(
        batch_size=batch_size,
        test_page=test_page,
        verbose=verbose
    )
    
    document_parse_node = DocumentParseNode(
        verbose=verbose,
        output_dir=output_dir,
        language=language,
        domain_type=domain_type
    )
    
    post_document_parse_node = PostDocumentParseNode(
        verbose=verbose,
        language=language,
        domain_type=domain_type
    )
    
    working_queue_node = WorkingQueueNode(verbose=verbose)

    # 첫 번째 워크플로우 생성
    workflow = StateGraph(ParseState)
    workflow.add_node("split_pdf_node", split_pdf_node)
    workflow.add_node("document_parse_node", document_parse_node)
    workflow.add_node("post_document_parse_node", post_document_parse_node)
    workflow.add_node("working_queue_node", working_queue_node)

    workflow.add_edge("split_pdf_node", "working_queue_node")
    workflow.add_conditional_edges(
        "working_queue_node",
        continue_parse,
        {True: "document_parse_node", False: "post_document_parse_node"},
    )
    workflow.add_edge("document_parse_node", "working_queue_node")
    workflow.set_entry_point("split_pdf_node")
    parser_graph = workflow.compile()

    # 후처리 노드들 생성
    create_elements_node = CreateElementsNode(verbose=verbose)
    export_markdown = ExportMarkdown(verbose=verbose)
    export_table_csv = ExportTableCSV(verbose=verbose)
    export_image = ExportImage(verbose=verbose)
    page_elements_extractor_node = PageElementsExtractorNode(verbose=verbose)
    image_entity_extractor_node = ImageEntityExtractorNode(
        verbose=verbose,
        language=language,
        domain_type=domain_type
    )
    table_entity_extractor_node = TableEntityExtractorNode(
        verbose=verbose,
        language=language,
        domain_type=domain_type
    )
    merge_entity_node = MergeEntityNode(verbose=verbose)
    reconstruct_elements_node = ReconstructElementsNode(verbose=verbose)
    # langchain_document_node = LangChainDocumentNode(
    #     verbose=verbose,
    #     splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0),
    #     language=language,
    #     domain_type=domain_type
    # )

    # 후처리 워크플로우 생성
    post_process_workflow = StateGraph(ParseState)

    # 노드 추가
    post_process_workflow.add_node("document_parse", parser_graph)
    post_process_workflow.add_node("create_elements_node", create_elements_node)
    post_process_workflow.add_node("export_image", export_image)
    post_process_workflow.add_node("export_markdown", export_markdown)
    post_process_workflow.add_node("export_table_csv", export_table_csv)
    post_process_workflow.add_node(
        "page_elements_extractor_node", page_elements_extractor_node
    )
    post_process_workflow.add_node(
        "image_entity_extractor_node", image_entity_extractor_node
    )
    post_process_workflow.add_node(
        "table_entity_extractor_node", table_entity_extractor_node
    )
    post_process_workflow.add_node("merge_entity_node", merge_entity_node)
    post_process_workflow.add_node(
        "reconstruct_elements_node", reconstruct_elements_node
    )
    # post_process_workflow.add_node("langchain_document_node", langchain_document_node)

    # 엣지 연결
    post_process_workflow.add_edge("document_parse", "create_elements_node")
    post_process_workflow.add_edge("create_elements_node", "export_image")
    post_process_workflow.add_edge("export_image", "export_markdown")
    post_process_workflow.add_edge("export_image", "export_table_csv")
    post_process_workflow.add_edge("export_image", "page_elements_extractor_node")
    post_process_workflow.add_edge(
        "page_elements_extractor_node", "image_entity_extractor_node"
    )
    post_process_workflow.add_edge(
        "page_elements_extractor_node", "table_entity_extractor_node"
    )
    post_process_workflow.add_edge("image_entity_extractor_node", "merge_entity_node")
    post_process_workflow.add_edge("export_markdown", END)
    post_process_workflow.add_edge("export_table_csv", END)
    post_process_workflow.add_edge("table_entity_extractor_node", "merge_entity_node")
    post_process_workflow.add_edge("merge_entity_node", "reconstruct_elements_node")
    # post_process_workflow.add_edge(
    #     "reconstruct_elements_node", "langchain_document_node"
    # )
    # post_process_workflow.add_edge("langchain_document_node", END)
    post_process_workflow.add_edge("reconstruct_elements_node", END)

    post_process_workflow.set_entry_point("document_parse")

    memory = MemorySaver()
    return post_process_workflow.compile(checkpointer=memory)