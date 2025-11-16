#!/usr/bin/env python3
"""
🔧 FIX GUI WINDOW FREEZING - Resolver travamento de janelas em múltiplas instâncias
"""

def analyze_gui_problems():
    """Analisar problemas identificados na GUI do RobotV7"""
    
    print("🔍 ANÁLISE DOS PROBLEMAS DE TRAVAMENTO DA GUI")
    print("=" * 60)
    
    print("🚨 PROBLEMAS IDENTIFICADOS:")
    print("1. Thread Safety Issues:")
    print("   - update_stats() roda a cada 1.5s sem verificar se GUI está responsiva")
    print("   - Múltiplas chamadas root.after() podem se acumular")
    print("   - _log() acessa GUI diretamente de threads diferentes")
    
    print("\n2. Visibility Heartbeat Conflicts:")
    print("   - _visibility_heartbeat() roda a cada 5s")
    print("   - _force_show_window() usa attributes('-topmost') que pode conflitar")
    print("   - Múltiplas instâncias competindo por foco")
    
    print("\n3. Resource Leaks:")
    print("   - Callbacks root.after() não são cancelados ao fechar")
    print("   - Event bindings acumulam sem cleanup")
    print("   - Threading sem proper cleanup")
    
    print("\n4. Windows-specific Issues:")
    print("   - Win32 API calls podem falhar silenciosamente")
    print("   - ShowWindow/BringWindowToTop podem conflitar entre instâncias")
    
    print("\n🔧 SOLUÇÕES PROPOSTAS:")
    print("1. Implementar queue thread-safe para logs")
    print("2. Adicionar cleanup de callbacks ao fechar")
    print("3. Melhorar gerenciamento de foco entre múltiplas instâncias")
    print("4. Adicionar verificação de responsividade da GUI")
    print("5. Implementar debouncing para operações de visibilidade")

def create_gui_fixes():
    """Criar correções para os problemas da GUI"""
    
    print("\n🔧 CRIANDO CORREÇÕES PARA GUI")
    print("=" * 60)
    
    # Correção 1: Thread-safe logging
    thread_safe_logging = '''
import queue
import threading

class ThreadSafeGUI:
    def __init__(self, root):
        self.root = root
        self.log_queue = queue.Queue()
        self.update_callbacks = []
        self.is_closing = False
        
        # Processar queue de logs de forma thread-safe
        self.process_log_queue()
    
    def enqueue_log(self, message):
        """Thread-safe method to add log messages"""
        if not self.is_closing:
            try:
                self.log_queue.put(message, timeout=0.1)
            except queue.Full:
                pass  # Drop message if queue is full
    
    def process_log_queue(self):
        """Process log messages from queue in main thread"""
        try:
            while not self.log_queue.empty():
                message = self.log_queue.get_nowait()
                if hasattr(self, 'log_text'):
                    self.log_text.insert(tk.END, message + "\\n")
                    self.log_text.see(tk.END)
        except queue.Empty:
            pass
        except Exception:
            pass
        
        # Reagendar se não estiver fechando
        if not self.is_closing:
            self.root.after(100, self.process_log_queue)
'''
    
    # Correção 2: Cleanup de callbacks
    cleanup_callbacks = '''
    def cleanup_callbacks(self):
        """Cancel all pending callbacks before closing"""
        self.is_closing = True
        
        # Cancel all registered callbacks
        for callback_id in self.update_callbacks:
            try:
                self.root.after_cancel(callback_id)
            except:
                pass
        
        self.update_callbacks.clear()
    
    def safe_after(self, delay, callback):
        """Thread-safe wrapper for root.after with cleanup tracking"""
        if not self.is_closing:
            callback_id = self.root.after(delay, callback)
            self.update_callbacks.append(callback_id)
            return callback_id
        return None
'''
    
    # Correção 3: Melhor gerenciamento de visibilidade
    visibility_management = '''
    def __init__(self, root):
        # ... existing init code ...
        self.last_visibility_check = 0
        self.visibility_debounce = 2.0  # 2 seconds debounce
        self.focus_lock = threading.Lock()
    
    def _smart_visibility_heartbeat(self):
        """Smarter visibility check with debouncing"""
        try:
            current_time = time.time()
            
            # Debounce - só verificar se passou tempo suficiente
            if current_time - self.last_visibility_check < self.visibility_debounce:
                self.safe_after(1000, self._smart_visibility_heartbeat)
                return
            
            self.last_visibility_check = current_time
            
            # Verificar se realmente precisa restaurar
            state = self.root.state()
            if state in ("iconic", "withdrawn"):
                with self.focus_lock:
                    self._gentle_restore_window()
            
        except Exception:
            pass
        
        # Reagendar com intervalo maior
        self.safe_after(3000, self._smart_visibility_heartbeat)
    
    def _gentle_restore_window(self):
        """Gentle window restoration without aggressive focus stealing"""
        try:
            # Só restaurar, não forçar foco
            if self.root.state() in ("iconic", "withdrawn"):
                self.root.deiconify()
                self.root.state('normal')
                
                # Não usar topmost - pode conflitar com outras instâncias
                self.root.lift()
                
        except Exception:
            pass
'''
    
    # Correção 4: Update stats melhorado
    improved_update_stats = '''
    def __init__(self, root):
        # ... existing init code ...
        self.last_stats_update = 0
        self.stats_update_interval = 2.0  # 2 seconds instead of 1.5
        self.gui_responsive = True
    
    def update_stats(self):
        """Improved stats update with responsiveness check"""
        try:
            current_time = time.time()
            
            # Skip update if too frequent or GUI not responsive
            if (current_time - self.last_stats_update < self.stats_update_interval or 
                not self.gui_responsive):
                self.safe_after(500, self.update_stats)
                return
            
            self.last_stats_update = current_time
            
            # Check GUI responsiveness
            start_time = time.time()
            self.root.update_idletasks()
            update_time = time.time() - start_time
            
            # If update takes too long, GUI might be freezing
            self.gui_responsive = update_time < 0.1
            
            if not self.gui_responsive:
                print(f"[WARNING] GUI responsiveness issue detected: {update_time:.3f}s")
                self.safe_after(5000, self.update_stats)  # Wait longer
                return
            
            # ... existing stats update code ...
            
        except Exception as e:
            print(f"[ERROR] Stats update failed: {e}")
        finally:
            # Schedule next update with adaptive interval
            interval = 3000 if not self.gui_responsive else 2000
            self.safe_after(interval, self.update_stats)
'''
    
    print("✅ Correções criadas:")
    print("1. Thread-safe logging com queue")
    print("2. Cleanup de callbacks")
    print("3. Gerenciamento inteligente de visibilidade")
    print("4. Update stats com verificação de responsividade")
    
    return {
        'thread_safe_logging': thread_safe_logging,
        'cleanup_callbacks': cleanup_callbacks,
        'visibility_management': visibility_management,
        'improved_update_stats': improved_update_stats
    }

if __name__ == "__main__":
    analyze_gui_problems()
    fixes = create_gui_fixes()
    
    print("\n🚀 PRÓXIMOS PASSOS:")
    print("1. Aplicar as correções no RobotV7.py")
    print("2. Testar com múltiplas instâncias")
    print("3. Monitorar responsividade das janelas")
    print("4. Ajustar intervalos se necessário")