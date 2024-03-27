from langchain.chains.query_constructor.base import AttributeInfo

procedures = [  ## 필터링
    AttributeInfo(
        name="category",
        description="Classification based on query intent. The intent of this field is '신청 절차 문의'",
        type="string",
    ),
    # AttributeInfo(
    #     name="정책명",
    #     description="The policy name related to passage",
    #     type="string",
    # ),
    # AttributeInfo(
    #     name="단어",
    #     description="The term explained by the passage",
    #     type="string",
    # ),
    AttributeInfo(
        name="단계",
        description="The step to apply for the policy",
        type="string",
    ),
]

qualifications = [
    AttributeInfo(
        name="category",
        description="Classification based on query intent. The intent of this field is '신청 자격 문의'",
        type="string",
    ),
    # AttributeInfo(
    #     name="정책명",
    #     description="The policy name related to passage",
    #     type="string",
    # ),
    AttributeInfo(
        name="신청 자격",
        description="Words that become keywords in queries inquiring about application qualifications",
        type="string",
    ),
    AttributeInfo(
        name="단계",
        description="Words that become keywords in queries inquiring about application qualifications",
        type="string",
    ),
]

# Intentions are one of three types: 신청 절차 문의, 신청 자격 문의, 단순 질의
simple_query = [
    AttributeInfo(
        name="category",
        description="Classification based on query intent. The intent of this field is '단순 질의'",
        type="string",
    ),
    AttributeInfo(
        name="질문",
        description="Case sentences with similar meaning to the query",
        type="string",
    ),
]