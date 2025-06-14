import React, { useState } from 'react'; 
import axios from 'axios'; 
import { useNavigate } from 'react-router-dom'; 
import './AuthPage.css'; 
const AuthPage = ({ handleLogin }) => { 
  const navigate = useNavigate();  
  const [username, setUsername] = useState(''); 
  const [password, setPassword] = useState(''); 
  const [isLogin, setIsLogin] = useState(true); 
  const [error, setError] = useState(''); 
 
  const handleLoginSubmit = async (e) => { 
    e.preventDefault(); 
    try { 
      const response = await axios.post(`http://localhost:5000/login`, { username, password }); 
      handleLogin(response.data.user); 
      navigate('/'); 
      console.log('Login successful!'); 
    } catch (error) { 
      if (error.response.status === 400) { 
        setError('Invalid Username or Password'); 
      } else { 
        setError('An error occurred during login'); 
      } 
    } 
  }; 
 
  const handleSignupSubmit = async (e) => { 
    e.preventDefault(); 
    try { 
      const response = await axios.post(`http://localhost:5000/signup`, { username, password }); 
      handleLogin(response.data.user); 
      window.location.reload(); 
      console.log('Signup successful!'); 
    } catch (error) { 
      if (error.response.status === 400) { 
        setError('User Already Exists'); 
      }else if(error.response.status===401){ 
        setError('Username not met requirements!!'); 
      } 
      else if(error.response.status===402){ 
        setError('Password must be Strong!!'); 
      } 
      else { 
        setError('An error occurred during signup'); 
      } 
    } 
  }; 
 
  const handleToggleAuth = () => { 
    setIsLogin(!isLogin); 
    setError(''); 
  }; 
 
  return ( 
    <div className="auth-container"> 
      <form onSubmit={isLogin ? handleLoginSubmit : handleSignupSubmit}> 
        <h2>{isLogin ? 'Login' : 'Signup'}</h2> 
        <div> 
          <label htmlFor="username">Username:</label> 
          <input 
            type="text" 
            id="username" 
            value={username} 
            onChange={(e) => setUsername(e.target.value)} 
          /> 
        </div> 
        <div> 
          <label htmlFor="password">Password:</label> 
          <input 
            type="password" 
            id="password" 
            value={password} 
            onChange={(e) => setPassword(e.target.value)} 
          /> 
        </div> 
        {error && <div className="error">{error}</div>} 
        <button type="submit">{isLogin ? 'Login' : 'Signup'}</button> 
        <p onClick={handleToggleAuth}> 
          {isLogin ? <p>Don't have an account? <strong>Sign up</strong> here.</p> : <p>Already have an account?<strong>Log in</strong> here.</p>} 
        </p> 
      </form> 
    </div> 
  ); 
}; 
export default AuthPage; 