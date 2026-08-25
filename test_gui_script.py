import sys
sys.path.append('src')
from gui import AmharicAIApp

try:
    app = AmharicAIApp()
    app.update()

    print("test_session_switch exists:", hasattr(app, 'test_session_switch'))
    if hasattr(app, 'test_session_switch'):
        app.test_session_var.set(True)
        app._toggle_test_session()
        
        app.test_count_entry.delete(0, 'end')
        app.test_count_entry.insert(0, "2")
        
        app._start_test_session()
        print("Session active:", app.session_active)
        print("Banner visible text:", app.session_label.cget("text"))
        
        print("Number of thumb buttons:", len(app.card_container.winfo_children()))
        first_thumb_btn = app.card_container.winfo_children()[0]
        first_thumb_btn.invoke()
        
    app.update()
    print("Test passed without hard crashes.")
except Exception as e:
    import traceback
    traceback.print_exc()
