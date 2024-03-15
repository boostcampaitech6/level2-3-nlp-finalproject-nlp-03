import React, { useState } from 'react';
import { Button, Container, Typography, Grid, TextField, MenuItem } from '@mui/material';
import LoadingPage from './LoadingPage'; // LoadingPage import

const SelectPage = () => {
    const [formData, setFormData] = useState({
        birthDate: '',
        occupation: 'skip',
        preferredPolicyArea: 'skip',
        personalIncomeRange: 'skip',
        householdIncomeRange: 'skip',
    });
    const [isLoading, setIsLoading] = useState(false); // isLoading 상태 추가

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData({
            ...formData,
            [name]: value
        });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true); // 폼 제출 시 로딩 표시
    };

    return (
        isLoading ? <LoadingPage formData={formData} /> :
        (
            <Container maxWidth="sm" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh' }}>
                <Container maxWidth="sm">
                    <Typography variant="h5" align="center" gutterBottom style={{ marginBottom: '20px' }}>
                        입력을 많이 할수록 더 정확한 추천이 가능합니다.
                    </Typography>
                    <Typography fontSize={'14px'} align="right" gutterBottom style={{ marginBottom: '20px' }}>
                        * 건너뛰고 싶은 항목은 입력하지 않으셔도 됩니다.
                    </Typography>
                    <form onSubmit={handleSubmit}>
                        <Grid container spacing={2}>
                            <Grid item xs={12}>
                                <TextField
                                    fullWidth
                                    label="출생년월일"
                                    type="date"
                                    InputLabelProps={{ max: "9999-12-31", shrink: true }}
                                    name="birthDate"
                                    value={formData.birthDate}
                                    onChange={handleChange}
                                    defaultValue="skip"
                                />
                            </Grid>
                            <Grid item xs={12}>
                                <TextField
                                    fullWidth
                                    select
                                    label="구직상태"
                                    name="occupation"
                                    value={formData.occupation}
                                    onChange={handleChange}
                                    defaultValue="skip"
                                >
                                    <MenuItem value="구직자">구직자</MenuItem>
                                    <MenuItem value="재직자">재직자</MenuItem>
                                    <MenuItem value="창업자">창업자</MenuItem>
                                    <MenuItem value="대학생">대학생</MenuItem>
                                    <MenuItem value="기타">기타</MenuItem>
                                    <MenuItem value="skip">건너뛰기</MenuItem>
                                </TextField>
                            </Grid>
                            <Grid item xs={12}>
                                <TextField
                                    fullWidth
                                    select
                                    label="선호하는 정책 분야"
                                    name="preferredPolicyArea"
                                    value={formData.preferredPolicyArea}
                                    onChange={handleChange}
                                    defaultValue="skip"
                                >
                                    <MenuItem value="일자리">일자리</MenuItem>
                                    <MenuItem value="주거">주거</MenuItem>
                                    <MenuItem value="교육">교육</MenuItem>
                                    <MenuItem value="복지문화">복지문화</MenuItem>
                                    <MenuItem value="참여권리">참여권리</MenuItem>
                                    <MenuItem value="skip">건너뛰기</MenuItem>
                                </TextField>
                            </Grid>
                            <Grid item xs={12}>
                                <TextField
                                    fullWidth
                                    select
                                    label="소득(세전 개인 소득), 단위: 만원"
                                    name="personalIncomeRange"
                                    value={formData.personalIncomeRange}
                                    onChange={handleChange}
                                    defaultValue="skip"
                                >
                                    <MenuItem value="0">없음</MenuItem>
                                    <MenuItem value="3000">3000 미만</MenuItem>
                                    <MenuItem value="4000">3000 ~ 4000</MenuItem>
                                    <MenuItem value="5000">4000 ~ 5000</MenuItem>
                                    <MenuItem value="6000">5000 ~ 6000</MenuItem>
                                    <MenuItem value="7000">6000 이상</MenuItem>
                                    <MenuItem value="skip">건너뛰기</MenuItem>
                                </TextField>
                            </Grid>
                            <Grid item xs={12}>
                                <TextField
                                    fullWidth
                                    select
                                    label="가구 소득, 단위: 만원"
                                    name="householdIncomeRange"
                                    value={formData.householdIncomeRange}
                                    onChange={handleChange}
                                    defaultValue="skip"
                                >
                                    <MenuItem value="0">없음</MenuItem>
                                    <MenuItem value="3000">3000 미만</MenuItem>
                                    <MenuItem value="4000">3000 ~ 4000</MenuItem>
                                    <MenuItem value="5000">4000 ~ 5000</MenuItem>
                                    <MenuItem value="6000">5000 ~ 6000</MenuItem>
                                    <MenuItem value="7000">6000 이상</MenuItem>
                                    <MenuItem value="skip">건너뛰기</MenuItem>
                                </TextField>
                            </Grid>
                            {/* 다른 입력 항목들도 여기에 추가 */}
                        </Grid>
                        <Button type="submit" variant="contained" color="primary" style={{ marginTop: '20px' }}>
                            완료
                        </Button>
                    </form>
                </Container>
            </Container>
        )
    );
};

export default SelectPage;
