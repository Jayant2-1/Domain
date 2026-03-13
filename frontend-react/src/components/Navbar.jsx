import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Brain, LogOut, LayoutDashboard } from 'lucide-react';

export default function Navbar() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  return (
    <nav className="navbar">
      <Link to="/" className="nav-brand">
        <Brain size={28} />
        <span>MLML</span>
      </Link>

      <div className="nav-links">
        {user ? (
          <>
            <Link to="/dashboard" className="nav-link">
              <LayoutDashboard size={16} />
              Dashboard
            </Link>
            <span className="nav-user">{user.username}</span>
            <button onClick={handleLogout} className="nav-btn-outline">
              <LogOut size={16} />
              Logout
            </button>
          </>
        ) : (
          <>
            <Link to="/login" className="nav-link">Log in</Link>
            <Link to="/register" className="nav-btn-primary">Get Started</Link>
          </>
        )}
      </div>
    </nav>
  );
}
