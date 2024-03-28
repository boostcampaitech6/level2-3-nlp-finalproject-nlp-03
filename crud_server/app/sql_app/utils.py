from datetime import datetime
from .schemas import PolicyInfoResponse, Policy, PolicyV2, PolicyV2InfoResponse


def calculate_age(birth_date: str) -> int:
    """
    Calculate age based on the given birth date.

    Parameters:
        birth_date (str): A string representing the birth date in the format 'YYYY-MM-DD'.

    Returns:
        int: The calculated age.
    """
    # Convert birth date string to a datetime object
    birth_date_obj = datetime.strptime(birth_date, "%Y-%m-%d")
    
    # Get the current date
    current_date = datetime.now()
    
    # Calculate age
    age = current_date.year - birth_date_obj.year
    
    # Adjust age if the birth month and day have not occurred yet this year
    if current_date.month < birth_date_obj.month or \
            (current_date.month == birth_date_obj.month and current_date.day < birth_date_obj.day):
        age -= 1
    
    return age


def map_policy_to_response(policy: Policy) -> PolicyInfoResponse:
    """
    Map a Policy object to a PolicyInfoResponse object.

    Parameters:
        policy (Policy): A Policy object.

    Returns:
        PolicyInfoResponse: A PolicyInfoResponse object mapped from the provided Policy object.
    """
    end_date = datetime.strptime(policy.end_date, "%Y-%m-%d")  # 문자열을 datetime으로 변환
    d_day = (end_date - datetime.now()).days  # d_day 계산

    return PolicyInfoResponse(
        id=policy.PolicyID,
        PolicyName=policy.PolicyName,
        d_day=d_day,
        policy_type=policy.policyType,
        org_name=policy.OrgName
    )

def map_policyV2_to_response(policy: PolicyV2) -> PolicyV2InfoResponse:
    """
    Map a Policy object to a PolicyInfoResponse object.

    Parameters:
        policy (Policy): A Policy object.

    Returns:
        PolicyInfoResponse: A PolicyInfoResponse object mapped from the provided Policy object.
    """

    return PolicyV2InfoResponse(
        id=policy.PolicyID,
        PolicyName=policy.PolicyName,
        d_day=policy.D_day,
        policy_type=policy.policyType,
        org_name=policy.OrgName,
        Progress=policy.Progress
    )