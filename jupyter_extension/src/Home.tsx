import * as React from 'react';
import Razer_logo from './Jupyter_AIKit_logo.svg';

const logoUrl =
  "data:image/svg+xml;utf8," + encodeURIComponent(Razer_logo);


export function Home(props: { open: (path: string) => void }) {
  React.useEffect(() => { 
    document.title = 'Razer AIKit';
    // Set the entire page background to black
    document.body.style.backgroundColor = '#000000';
    const mainArea = document.querySelector('.jp-MainAreaWidget');
    if (mainArea) {
      (mainArea as HTMLElement).style.backgroundColor = '#000000';
    }
    
    return () => {
      // Cleanup: reset to default when component unmounts
      document.body.style.backgroundColor = '';
      const mainArea = document.querySelector('.jp-MainAreaWidget');
      if (mainArea) {
        (mainArea as HTMLElement).style.backgroundColor = '';
      }
    };
  }, []);
  
  const containerStyle: React.CSSProperties = {
    minHeight: '100vh',
    background: '#000000',
    padding: '16px 64px 48px 64px',
    color: '#ffffff',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    margin: '0 auto',
    overflowY: 'auto',
    boxSizing: 'border-box'
  };

  const contentWrapperStyle: React.CSSProperties = {
    width: '100%',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-start',
    flex: '1 1 auto',
    overflowY: 'auto'
  };

  const logoStyle: React.CSSProperties = {
    width: '800px',
    marginTop: '0',
    marginBottom: '16px'
  };

  const gridContainerStyle: React.CSSProperties = {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '24px',
    width: '100%',
    marginBottom: '24px'
  };

  const sectionStyle: React.CSSProperties = {
    width: '100%'
  };

  const integrationsStyle: React.CSSProperties = {
    width: '49%'
  };

  const sectionTitleStyle: React.CSSProperties = {
    color: '#ffffff',
    marginTop: 0,
    marginBottom: '16px',
    fontWeight: 500,
    textTransform: 'uppercase',
    letterSpacing: '1px',
    fontSize: '16px',
    textAlign: 'left'
  };

  const listStyle: React.CSSProperties = {
    listStyle: 'none',
    padding: 0,
    margin: 0
  };

  const linkStyle: React.CSSProperties = {
    color: '#ffffff',
    cursor: 'pointer',
    textDecoration: 'none',
    display: 'block',
    padding: '14px 16px',
    marginBottom: '8px',
    background: '#111111',
    border: '1px solid #222222',
    borderRadius: '4px',
    transition: 'all 0.2s ease',
    fontSize: '16px'
  };

  return (
    <div style={containerStyle}>
      <img 
        src={logoUrl} 
        alt="Razer AIKit" 
        style={logoStyle}
      />
      <div style={{
        border: '2px solid #44D62C',
        borderRadius: '8px',
        padding: '15px',
        margin: '10px 0 40px 0',
        boxShadow: '0 2px 4px rgba(68, 214, 44, 0.2)',
        backgroundColor: 'transparent'
      }}>
        <p style={{ margin: 0, color: '#ffffff', lineHeight: '1.6' , fontSize: '16px'}}>
          Razer AIKit is open-source AI development toolkit built for engineers and researchers. 
          Designed for easy, out-of-the-box setup, AIKit delivers cloud-grade GPU acceleration and scalability directly on your desktop. 
          It empowers users to harness the power of large language models (LLMs) while maintaining data privacy and minimizing latency through local or distributed computation.
          <br /><br />
          These notebooks will guide you through the process of setting up and using Razer AIKit for various AI tasks,
          including inferencing, fine-tuning, and integrating with popular APIs.
        </p>
      </div>
      <div style={contentWrapperStyle}>
        <div style={gridContainerStyle}>
        <div style={sectionStyle}>
          <h2 style={sectionTitleStyle}>Inferencing: Model deployment</h2>
          <ul style={listStyle}>
            <li>
              <a 
                style={linkStyle}
                onClick={() => props.open('notebooks/1_On_Device_Inferencing.ipynb')}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = '#44d62c';
                  e.currentTarget.style.background = '#151515';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = '#222222';
                  e.currentTarget.style.background = '#111111';
                }}
              >
                💻 1. On-Device Inferencing
              </a>
            </li>
            <li>
              <a 
                style={linkStyle}
                onClick={() => props.open('notebooks/2a_(Head)_Distributed_Inferencing.ipynb')}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = '#44d62c';
                  e.currentTarget.style.background = '#151515';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = '#222222';
                  e.currentTarget.style.background = '#111111';
                }}
              >
                🌐 2a. Distributed Inferencing (Head)
              </a>
            </li>
            <li>
              <a 
                style={linkStyle}
                onClick={() => props.open('notebooks/2b_(Node)_Distributed_Inferencing.ipynb')}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = '#44d62c';
                  e.currentTarget.style.background = '#151515';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = '#222222';
                  e.currentTarget.style.background = '#111111';
                }}
              >
                🌐 2b. Distributed Inferencing (Node)
              </a>
            </li>
            <li>
              <a 
                style={linkStyle}
                onClick={() => props.open('notebooks/3_On_Device_Image_Generation.ipynb')}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = '#44d62c';
                  e.currentTarget.style.background = '#151515';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = '#222222';
                  e.currentTarget.style.background = '#111111';
                }}
              >
                🎨 3. On-Device Image Generation
              </a>
            </li>
            <li>
              <a 
                style={linkStyle}
                onClick={() => props.open('notebooks/4_On_Device_Audio_Generation.ipynb')}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = '#44d62c';
                  e.currentTarget.style.background = '#151515';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = '#222222';
                  e.currentTarget.style.background = '#111111';
                }}
              >
                🎵 4. On-Device Audio Generation
              </a>
            </li>
            <li>
              <a 
                style={linkStyle}
                onClick={() => props.open('notebooks/5_On_Device_Video_Generation.ipynb')}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = '#44d62c';
                  e.currentTarget.style.background = '#151515';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = '#222222';
                  e.currentTarget.style.background = '#111111';
                }}
              >
                🎬 5. On-Device Video Generation
              </a>
            </li>
          </ul>
        </div>

        <div style={sectionStyle}>
          <h2 style={sectionTitleStyle}>Fine-Tuning: Model customization</h2>
          <ul style={listStyle}>
            <li>
              <a 
                style={linkStyle}
                onClick={() => props.open('notebooks/5_On_Device_Fine_Tuning_LoRA.ipynb')}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = '#44d62c';
                  e.currentTarget.style.background = '#151515';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = '#222222';
                  e.currentTarget.style.background = '#111111';
                }}
              >
                💻 5. On-Device Fine-Tuning (LoRA)
              </a>
            </li>
            <li>
              <a 
                style={linkStyle}
                onClick={() => props.open('notebooks/6a_(Head)_Distributed_Fine_Tuning_LoRA.ipynb')}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = '#44d62c';
                  e.currentTarget.style.background = '#151515';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = '#222222';
                  e.currentTarget.style.background = '#111111';
                }}
              >
                🌐 6a. Distributed Fine-Tuning LoRA (Head)
              </a>
            </li>
            <li>
              <a 
                style={linkStyle}
                onClick={() => props.open('notebooks/6b_(Node)_Distributed_Fine_Tuning_LoRA.ipynb')}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = '#44d62c';
                  e.currentTarget.style.background = '#151515';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = '#222222';
                  e.currentTarget.style.background = '#111111';
                }}
              >
                🌐 6b. Distributed Fine-Tuning LoRA (Node)
              </a>
            </li>
          </ul>
        </div>
      </div>

      <div style={integrationsStyle}>
        <h2 style={sectionTitleStyle}>Integrations: Additional usage</h2>
        <ul style={listStyle}>
          <li>
            <a 
              style={linkStyle}
              onClick={() => props.open('notebooks/7_Integrating_Razer_AIKit_with_OpenAI_API.ipynb')}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = '#44d62c';
                e.currentTarget.style.background = '#151515';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = '#222222';
                e.currentTarget.style.background = '#111111';
              }}
            >
              7. Integrating AIKit with OpenAI API
            </a>
          </li>
          <li>
            <a 
              style={linkStyle}
              onClick={() => props.open('notebooks/8_Semantic_Search.ipynb')}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = '#44d62c';
                e.currentTarget.style.background = '#151515';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = '#222222';
                e.currentTarget.style.background = '#111111';
              }}
            >
              8. Semantic Search
            </a>
          </li>
        </ul>
      </div>
    </div>
    </div>
  );
}
