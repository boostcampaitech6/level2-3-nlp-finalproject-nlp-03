import React, { useState } from 'react';
import { Paper, Typography, IconButton, Button } from '@mui/material';
import { ExpandMore, ExpandLess } from '@mui/icons-material';

const SourceDocument = ({ text }) => {
  const [expanded, setExpanded] = useState(false);

  const handleToggle = () => {
    setExpanded(!expanded);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', marginBottom: '10px', paddingBottom: '10px' }}>
      <div style={{ justifyContent:'flex-end', textAlign: 'right', cursor: 'pointer', color: '#1976D2', marginBottom: '5px' }} onClick={handleToggle}>
        {expanded ? (
          <>
            <Button>접기</Button>
          </>
        ) : (
          <>
            <Button>더보기</Button>
          </>
        )}
      </div>
      { expanded ? (
        <>
          <Paper elevation={0} sx={{ border: 1, borderRadius: 2, borderColor: 'grey.300' }} style={{ padding: '10px', maxWidth: '100%', backgroundColor: '#FFFFFF' }}>
            <h4 style={{textAlign: 'left', margin:'20px' }}> 참고 자료 </h4>
            <br></br>
            <Typography variant="body1" style={{ margin:'20px' ,textAlign: 'left', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: expanded ? 'normal' : 'nowrap' }}>
              {text}
            </Typography>
          </Paper>
        </>
      )
      : null
      }   
    </div>
  );
};

export default SourceDocument;