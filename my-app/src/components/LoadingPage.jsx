import React, { useState, useEffect } from 'react';
import CircularProgress from '@mui/material/CircularProgress';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import { useNavigate } from 'react-router-dom';

const LoadingPage = ({ formData }) => {
    const [progress, setProgress] = useState(0);
    const navigate = useNavigate();

    useEffect(() => {
        // formData 를 담아 서버로 보내고 추천 결과를 받아오는 로직을 구현해야 합니다.

        const timer = setInterval(() => {
            setProgress((prevProgress) => (prevProgress >= 100 ? 0 : prevProgress + 10));
        }, 800);

        setTimeout(() => {
            clearInterval(timer);
            navigate('/recommendPage'); // 프로그레스 바가 한 바퀴를 돌면 /recommendPage로 이동
        }, 8000); // 8000 밀리초 = 8초

        return () => {
            clearInterval(timer);
        };
    }, [navigate]);

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
            <Typography variant="h5" gutterBottom style={{ marginBottom: '20px' }}>
                맞춤형 정책을 찾고 있어요! 잠시만 기다려주세요.
            </Typography>
            <CircularProgress variant="determinate" value={progress} size={80} />
        </Box>
    );
};

export default LoadingPage;
