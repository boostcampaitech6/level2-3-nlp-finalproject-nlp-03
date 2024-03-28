import React, { useState } from 'react';
import { Button, Container, Typography, Grid, CardMedia } from '@mui/material';
import { Link } from 'react-router-dom';

const OnBoardingStartPage = () => {
    return (
        <Container maxWidth="md" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh' }}>
            <Container maxWidth="md">
                <Typography variant="h5" align="center" gutterBottom style={{ marginBottom: '20px' }}>
                    나에게 딱 맞는 청년 정책을 추천받고
                    <br />
                    <span style={{ color: '#1976d2', fontWeight: 'bold' }}>AI 상담 서비스</span>를 이용해보세요.
                </Typography>
                <Typography variant="h6" align="center" gutterBottom style={{ marginBottom: '50px'}}>
                    예상 소요 시간 <span style={{ color: '#1976d2' }}>30초</span>
                    <br />
                    총 문항 수 <span style={{ color: '#1976d2' }}>7문항</span>
                </Typography>
                <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '100px'}}>
                    <CardMedia
                        component="img"
                        src="/openai.png"
                        alt="OpenAI"
                        style={{ width: '30%', height: '30%' }}
                    />
                </div>
                <Grid container justifyContent="center">
                    <Grid item xs={12} md={6}>
                        <Link to="/select" style={{ textDecoration: 'none' }}>
                            <Button variant="contained" color="primary" style={{ width: '100%', height: '60px', fontSize: '20px'}}>
                                시작하기
                            </Button>
                        </Link>
                    </Grid>
                </Grid>
            </Container>
        </Container>
    );
};

export default OnBoardingStartPage;
