import React, { useState, useEffect, useRef } from 'react';
import { Container, Typography, TextField, Button, Grid } from '@mui/material';
import Message from './Message';
import SourceDocument from './SourceDocument';
import { useLocation } from 'react-router-dom';

const ChatAppV2 = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isResponse, setIsResponse] = useState(true);
  const messagesEndRef = useRef(null);
  const location = useLocation();
  const policy = location.state?.policy || {};
  const [chatbotType, setChatbotType] = useState('');

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (chatbotType == '신청자격') {
      setMessages([...messages, { text: `
안녕하세요. 저는 청년 정책 챗봇 길벗이라고 해요. 여기서는 ${policy.PolicyName}에 대한 신청자격상담이 가능합니다.\n
${policy.PolicyName}은 전세자금이 부족한 청년들에게 청년전용 버팀목 전세자금을 대출해주는 정책으로, 신청 자격은 다음과 같습니다.
                                  `, role: 'chatbot' },
                                { text: `
1. (계약) 주택임대차계약을 체결하고 임차보증금의 5% 이상을 지불한 자
2. (세대주) 대출접수일 현재 만 19세 이상 만 34세 이하의 세대주 (예비 세대주 포함) 

※ 단, 쉐어하우스(채권양도협약기관 소유주택에 한함)에 입주하는 경우 예외적으로 세대주 요건을 만족하지 않아도 이용 가능

3. (무주택) 세대주를 포함한 세대원 전원이 무주택인 자
4. (중복대출 금지) 주택도시기금 대출, 은행재원 전세자금 대출 및 주택담보 대출 미이용자
5. (소득) 대출신청인과 배우자의 합산 총소득이 5천만원 이하인 자 

※ 단, 혁신도시 이전 공공기관 종사자, 타 지역으로 이주하는 재개발 구역내 세입자, 다자녀가구, 2자녀 가구인 경우 6천만원 이하, 신혼가구인 경우는 7.5천만원 이하인 자

6. (자산) 대출신청인 및 배우자의 합산 순자산 가액이 통계청에서 발표하는 최근년도 가계금융복지조사의 ‘소득 5분위별 자산 및 부채현황’ 중 소득 3분위 전체가구 평균값 이하(십만원 단위에서 반올림)인 자

※ 2024년도 기준 3.45억원

7. (신용도) 아래 요건을 모두 충족하는 자

신청인(연대입보한 경우 연대보증인 포함)이 한국신용정보원 “신용정보관리규약”에서 정하는 아래의 신용정보 및 해제 정보가 남아있는 경우 대출 불가능

1. 연체, 대위변제·대지급, 부도, 관련인 정보
2. 금융질서문란정보, 공공기록정보, 특수기록정보
3. 신용회복지원등록정보

그 외, 부부에 대하여 대출취급기관 내규로 대출을 제한하고 있는 경우에는 대출 불가능
                                        `, role: 'source', init: chatbotType },
        { text: `
관련하여 궁금하신 점이 있다면 다음과 같이 질문해 주세요.\n
예) ‘버팀목 소득 조건 확인은 어떻게 해?’, ‘버팀목 신용도 확인은 어디서 할 수 있어?’\n
*청년 정책 봇 길벗은 아직 열심히 공부 중이에요! 혹시 부족한 부분이 있다면 동행팀(seok.sukyung89@gmail.com)으로 연락 부탁드릴게요. 멍멍!
                                          `, role: 'chatbot' }]);
    }
    else if (chatbotType == '신청절차') {
      setMessages([...messages, { text: `
안녕하세요. 저는 청년 정책 챗봇 길벗이라고 해요. 여기서는  ${policy.PolicyName}에 대한 신청절차상담이 가능합니다. \n
${policy.PolicyName}은 전세자금이 부족한 청년들에게 청년전용 버팀목 전세자금을 대출해주는 정책으로, 신청 절차는 다음과 같습니다.
                                  `, role: 'chatbot' },
                                { text: `
1. 대출조건 확인 
2. 주택 가계약 
3. 대출신청 
4. 자산심사(HUG) 
5. 자산심사 결과 정보 송신(HUG) 
6. 서류제출 및 추가심사 진행(수탁은행) 
7. 대출승인 및 실행
                                        `, role: 'source', init: chatbotType },
      { text: `
관련하여 궁금하신 점이 있다면 다음과 같이 질문해 주세요.\n
예) ‘버팀목 대출신청 방법 알려 줘’, ‘자산심사는 뭐야?’\n
*청년 정책 봇 길벗은 아직 열심히 공부 중이에요! 혹시 부족한 부분이 있다면 동행팀(seok.sukyung89@gmail.com)으로 연락 부탁드릴게요. 멍멍!
                                        `, role: 'chatbot' }
                                      ]);
    }
    else if (chatbotType == '정보제공') {
      setMessages([...messages, { text: `
안녕하세요. 저는 청년 정책 챗봇 길벗이라고 해요. 여기서는  ${policy.PolicyName}에 대한 단순 문의가 가능합니다.\n
${policy.PolicyName}은 전세자금이 부족한 청년들에게 청년전용 버팀목 전세자금을 대출해주는 정책으로, 지원내용은 다음과 같습니다.
                                        `, role: 'chatbot' },
                                      { text: `
□ 대출금리 : 연 1.8~2.7%
※20.5.8~20.8.9까지 접수된 청년가구 우대금리(0.6%p) 부여 계좌는 변경 전 기준금리(소득구간별 1.8%∼2.4%)가 적용되며, 연장 시점에 변경된 기준금리가 적용

□ 대출한도
○ 호당대출한도: 2억원 이하(단, 만25세미만 단독세대주인 경우 1.5억원 이하)
○ 소요자금에 대한 대출비율

- 신규계약 : 전세금액의 80% 이내
- 갱신계약 : 증액금액 이내에서 증액 후 총 보증금의 80% 이내

□ 대출기간 : 최초 2년(4회연장, 최장 10년 이용가능)
○ 주택도시보증공사 전세금안심대출 보증서 : 최대 2년 1개월(4회 연장하여 최장 10년 5개월 가능)
○ 최장 10년 이용 후 연장시점 기준 미성년 1자녀당 2년 추가(최장 20년 이용 가능)
                                              `, role: 'source', init: chatbotType },
            { text: `
관련하여 궁금하신 점이 있다면 다음과 같이 질문해 주세요.\n
예) ‘버팀목 운영 기관이 어디야?’, ‘버팀목의 관련 참고 사이트 알려 줘’ \n
*청년 정책 봇 길벗은 아직 열심히 공부 중이에요! 혹시 부족한 부분이 있다면 동행팀(seok.sukyung89@gmail.com)으로 연락 부탁드릴게요. 멍멍!
                                              `, role: 'chatbot' }]);
    }
  }, [chatbotType]);

  const handleMessageSend = async () => {
    if (inputValue.trim() !== '') {
      // 사용자가 보낸 메시지를 추가
      setMessages([...messages, { text: inputValue, role: 'user' }]);
      setInputValue('');
      // 서버 요청 보내고 응답 받기 (시뮬레이션)
      try {
        const response = await fetch('http://localhost:8000/query', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            "query": inputValue // inputValue를 그대로 보냅니다.
          }) // inputValue를 그대로 보냅니다.
        });
        const data = await response.json();

        // 서버로부터 받은 응답을 채팅창에 추가
        setMessages(messages => [
          ...messages, 
          { text: data.response, role: 'chatbot' },
          { text: data.source_document, role: 'source', init: ''}
        ]);
        console.log(messages)
      } catch (error) {
        console.error('Error sending message:', error);
      }
      setIsResponse(true);
      setInputValue('');
    }
  };

  const handleButtonClick = (type) => {
    setChatbotType(type);
  };

  const handleInputChange = (e) => {
    if (isResponse == false) {
      return
    }
    setInputValue(e.target.value);
  };

  const handleKeyDown = (e) => {
    if (e.nativeEvent.isComposing) return
    if (e.key === 'Enter') {
      console.log('chatbotType:', chatbotType);
      console.log('policy:', policy);
      setIsResponse(false);
      handleMessageSend();
    }
  };

  return (
    <Container maxWidth="md" style={{ marginTop: '50px' }}>
      <Typography variant="h4" component="h1" align="center" gutterBottom>
        👨‍👩‍👧‍👦 {policy.PolicyName} 안내 챗봇
      </Typography>
      <Grid container spacing={2}>
        {/* chatbotType이 빈 문자열이 아닌 경우 */}
        {chatbotType !== '' ? (
          <>
            <Grid item xs={12}>
              <div style={{ minHeight: '80vh', maxHeight: '80vh', overflowY: 'auto' }}>
                {messages.map((message, index) => (
                    message.role === 'source' ? (
                      <SourceDocument key={index} text={message.text} init={message.init} />
                    ) : (
                      <Message key={index} text={message.text} role={message.role} />
                    )
                  ))}
                <div style={{ float: 'right', width: '8px', height: '100%', marginRight: '-8px' }} />
                <div ref={messagesEndRef} />
              </div>
            </Grid>
            <Grid item xs={12}>
              <Grid container spacing={2} alignItems="flex-end" justifyContent="flex-end">
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="질문을 입력해주세요."
                    variant="outlined"
                    value={inputValue}
                    onKeyDown={handleKeyDown}
                    onChange={handleInputChange}
                  />
                </Grid> 
              </Grid>
            </Grid>
          </>
        ) : (
          <Grid container spacing={2} justifyContent="center">
            <div style={{ paddingTop: '60px', minHeight: '60vh', maxHeight: '60vh', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
              <div style={{ marginBottom: '50px' }}>
                <Button
                  style={{
                    width: '200px',
                    height: '50px',
                    fontSize: '1rem',
                    fontWeight: 'bold',
                    borderRadius: '8px',
                    backgroundColor: chatbotType === '신청절차' ? '#1976D2' : '#2196F3',
                    color: '#FFFFFF',
                    border: 'none',
                    cursor: 'pointer',
                    transition: 'background-color 0.3s',
                  }}
                  onClick={() => handleButtonClick('신청절차')}
                >
                  신청절차 알아보기
                </Button>
              </div>
              <div style={{ marginBottom: '50px' }}>
                <Button
                  style={{
                    width: '200px',
                    height: '50px',
                    fontSize: '1rem',
                    fontWeight: 'bold',
                    borderRadius: '8px',
                    backgroundColor: chatbotType === '신청자격' ? '#388E3C' : '#4CAF50',
                    color: '#FFFFFF',
                    border: 'none',
                    cursor: 'pointer',
                    transition: 'background-color 0.3s',
                  }}
                  onClick={() => handleButtonClick('신청자격')}
                >
                  신청자격 확인하기
                </Button>
              </div>
              <div>
                <Button
                  style={{
                    width: '200px',
                    height: '50px',
                    fontSize: '1rem',
                    fontWeight: 'bold',
                    borderRadius: '8px',
                    backgroundColor: chatbotType === '정보제공' ? '#FFA000' : '#FFC107',
                    color: '#FFFFFF',
                    border: 'none',
                    cursor: 'pointer',
                    transition: 'background-color 0.3s',
                  }}
                  onClick={() => handleButtonClick('정보제공')}
                >
                  정보 제공받기
                </Button>
              </div>
            </div>
          </Grid>
        )}
      </Grid>
    </Container>
  );
};

export default ChatAppV2;
