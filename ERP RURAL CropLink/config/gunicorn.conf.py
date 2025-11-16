"""
Configuração do Gunicorn para CropLink
Otimizado para produção no Render
"""
import os
import multiprocessing

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
backlog = 2048

# Worker processes
workers = int(os.getenv('WEB_CONCURRENCY', multiprocessing.cpu_count() * 2 + 1))
worker_class = "sync"
worker_connections = 1000
timeout = int(os.getenv('WEB_TIMEOUT', '120'))
keepalive = 2

# Restart workers after this many requests, with some jitter
max_requests = 1000
max_requests_jitter = 100

# Preload the application before forking workers
preload_app = True

# Restart workers gracefully on code changes
reload = os.getenv('FLASK_ENV') == 'development'

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = os.getenv('LOG_LEVEL', 'info')
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "croplink"

# Server mechanics
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None
umask = 0

# SSL (usually handled by reverse proxy in production)
keyfile = None
certfile = None

# Security
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190

def when_ready(server):
    """Called just after the server is started."""
    server.log.info("CropLink server is ready. Listening on: %s", server.address)

def worker_int(worker):
    """Called just after a worker has been interrupted by SIGINT or SIGQUIT."""
    worker.log.info("Worker received INT or QUIT signal")

def pre_fork(server, worker):
    """Called just before a worker is forked."""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_worker_init(worker):
    """Called just after a worker has initialized the application."""
    worker.log.info("Worker initialized (pid: %s)", worker.pid)

def worker_abort(worker):
    """Called when a worker received the SIGABRT signal."""
    worker.log.info("Worker aborted (pid: %s)", worker.pid)

def pre_exec(server):
    """Called just before a new master process is forked."""
    server.log.info("Forked child, reexecuting")

def pre_request(worker, req):
    """Called just before a worker processes the request."""
    worker.log.debug("%s %s", req.method, req.path)

def post_request(worker, req, environ, resp):
    """Called after a worker processes the request."""
    worker.log.debug("Response: %s", resp.status)

# Error handling
def worker_exit(server, worker):
    """Called when a worker is exiting gracefully."""
    server.log.info("Worker exiting (pid: %s)", worker.pid)

# Environment-specific configurations
if os.getenv('FLASK_ENV') == 'production':
    # Production settings
    workers = min(int(os.getenv('WEB_CONCURRENCY', multiprocessing.cpu_count() * 2 + 1)), 4)
    timeout = 60
    keepalive = 5
    max_requests = 2000
    preload_app = True
    
elif os.getenv('FLASK_ENV') == 'development':
    # Development settings
    workers = 1
    timeout = 0  # No timeout in development
    reload = True
    preload_app = False
    loglevel = 'debug'

# Memory optimization
def on_starting(server):
    """Called just before the master process is initialized."""
    server.log.info("Starting CropLink with %d workers", workers)

def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    server.log.info("Reloading CropLink")

def on_exit(server):
    """Called just before the master process exits."""
    server.log.info("Shutting down CropLink")

# Custom error pages
default_proc_name = "croplink"

# Render-specific optimizations
if os.getenv('RENDER'):
    # Render platform specific settings
    bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"
    workers = 2  # Render free tier has memory limits
    worker_class = "sync"
    timeout = 120
    keepalive = 2
    max_requests = 500
    max_requests_jitter = 50
    preload_app = True
    
    # Render uses HTTP/1.1 proxy
    forwarded_allow_ips = "*"
    secure_scheme_headers = {
        'X-FORWARDED-PROTO': 'https',
    }
