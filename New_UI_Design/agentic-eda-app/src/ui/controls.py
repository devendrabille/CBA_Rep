from streamlit import st

def create_button(label, key=None, on_click=None, args=None, disabled=False):
    return st.button(label, key=key, on_click=on_click, args=args, disabled=disabled)

def create_slider(label, min_value, max_value, value=None, step=1, key=None):
    return st.slider(label, min_value=min_value, max_value=max_value, value=value, step=step, key=key)

def create_selectbox(label, options, index=0, key=None):
    return st.selectbox(label, options, index=index, key=key)

def create_text_input(label, value="", key=None):
    return st.text_input(label, value=value, key=key)

def create_checkbox(label, value=False, key=None):
    return st.checkbox(label, value=value, key=key)

def create_radio(label, options, index=0, key=None):
    return st.radio(label, options, index=index, key=key)

def create_multiselect(label, options, default=None, key=None):
    return st.multiselect(label, options, default=default, key=key)

def create_expander(label, expanded=False):
    return st.expander(label, expanded=expanded)