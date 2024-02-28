import React from 'react';
import { Paper, Typography } from '@mui/material';

const Message = ({ text, role }) => {
  const isUser = role === 'user';

  return (
    <div style={{ display: 'flex', justifyContent: isUser ? 'flex-end' : 'flex-start', marginBottom: '10px' }}>
      <Paper elevation={3} style={{ padding: '10px', maxWidth: '70%', backgroundColor: isUser ? '#DCF8C6' : '#E3F2FD' }}>
        <Typography variant="body1">{text}</Typography>
      </Paper>
    </div>
  );
};

export default Message;
