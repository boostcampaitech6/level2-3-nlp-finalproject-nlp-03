import React, { useState, useEffect } from 'react';
import CircularProgress from '@mui/material/CircularProgress';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import { useNavigate } from 'react-router-dom';

const LoadingPage = ({ formData }) => {
    const [progress, setProgress] = useState(0);
    const navigate = useNavigate();
    const policiesData = [];

    useEffect(() => {
        const fetchData = async () => {
            try {
                console.log(formData)
                const response = await fetch('http://110.165.18.177:8001/filter_policy_v3', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(formData)
                });

                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }

                const data = await response.json();
                console.log(data); // 받은 데이터를 확인하기 위해 콘솔에 출력

                clearInterval(timer);
                navigate('/recommendPage', { state: { policiesData: data } });
            } catch (error) {
                console.error('There was an error!', error);
                clearInterval(timer);
            }
        };

        const timer = setInterval(() => {
            setProgress((prevProgress) => (prevProgress >= 100 ? 0 : prevProgress + 10));
        }, 800);

        fetchData();

        return () => {
            clearInterval(timer);
        };
    }, [formData, navigate]);

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
