import React from 'react';
import { Container, Typography, Box } from '@mui/material';
import PolicyCard from './PolicyCard';
import { useLocation } from 'react-router-dom';

const RecommendPage = () => {
  const location = useLocation();
  const policiesData = location.state?.policiesData || [];

  return (
    <Container maxWidth="md" sx={{ marginTop: 4, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '100vh' }}>
      <Typography variant="h4" gutterBottom>
        추천 정책
      </Typography>
      <Box sx={{ display: 'flex', flexWrap: 'wrap'}}>
        {policiesData.map((policy, index) => (
          <Box key={index} sx={{ width: '45%', padding: 1, marginLeft: '15px' }}>
            <PolicyCard policy={policy} />
          </Box>
        ))}
      </Box>
    </Container>
  );
};

export default RecommendPage;
