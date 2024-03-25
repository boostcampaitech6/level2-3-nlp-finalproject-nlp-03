import React from 'react';
import { Card, CardContent, Typography, Button, Chip } from '@mui/material';
import { useNavigate } from 'react-router-dom';

const PolicyCard = ({ policy }) => {
  const navigate = useNavigate();
  const handleNavigateToChat = () => {
    // AI 상담 버튼을 클릭할 때 ChatAppV2로 이동합니다.
    console.log("채팅창으로 이동 : ", policy.policyID)
    navigate('/chatv2', { state: { policy: policy } }); 
  };

  return (
    <div style={{ height: '100%', maxHeight: '400px', width: '100%', maxWidth: '500px', margin: '10px auto' }}>
      <Card elevation={0} sx={{ border: 1, borderRadius: 2, borderColor: 'grey.300' }}>
        <CardContent>
          <Typography variant="h5" component="h5">
            {policy.PolicyName}
          </Typography>
          {/* PolicyName 이외의 속성들을 해시태그로 표시 */}
          <div style={{ marginTop: '8px' }}>
            {/* policy_type을 해시태그로 표시 */}
            <Chip label={`#${policy.policy_type}`} variant="outlined" style={{ marginRight: '4px', marginBottom: '4px' }} />
            {/* org_name을 해시태그로 표시 */}
            <Chip label={`#${policy.org_name}`} variant="outlined" style={{ marginRight: '4px', marginBottom: '4px' }} />
            {/* progress을 해시태그로 표시 */}
            <Chip label={`#${policy.Progress}`} variant="outlined" style={{ marginRight: '4px', marginBottom: '4px' }} />
            {/* d_day를 해시태그로 표시 */}
            {policy.Progress === "진행중" && (
              <Chip label={`#D-${policy.d_day}`} variant="outlined" style={{ marginRight: '4px', marginBottom: '4px' }} />
            )}
          </div>
          <Button variant="contained" onClick={handleNavigateToChat}>
            AI 상담 하기
          </Button>
        </CardContent>
      </Card>
    </div>
  );
};

export default PolicyCard;
