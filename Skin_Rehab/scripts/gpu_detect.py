import pyglet
from pyglet.gl import *
import ctypes

# Create a hidden window to initialize the OpenGL context
# (This is required before asking the driver for info)
try:
    window = pyglet.window.Window(visible=False)
except Exception as e:
    print(f"❌ Error creating window: {e}")
    exit()

def get_gl_string(name):
    # Get the raw C-pointer from OpenGL
    ptr = glGetString(name)
    # Cast it to a python string
    return ctypes.cast(ptr, ctypes.c_char_p).value.decode('utf-8')

print("="*40)
print("🕵️ GPU DETECTIVE")
print("="*40)

try:
    renderer = get_gl_string(GL_RENDERER)
    vendor = get_gl_string(GL_VENDOR)
    version = get_gl_string(GL_VERSION)

    print(f"👉 ACTIVE GPU:    {renderer}")
    print(f"👉 MANUFACTURER:  {vendor}")
    print(f"👉 OPENGL VERSION:{version}")
    print("="*40)

    if "NVIDIA" in vendor or "NVIDIA" in renderer:
        print("✅ SUCCESS: You are using the RTX Card!")
    elif "AMD" in vendor or "Intel" in vendor:
        print("❌ PROBLEM: You are using the Integrated Graphics (Slow).")
        print("   Action: Go to Windows Graphics Settings and force 'High Performance'")
        print(r"   for: C:\Users\Anuka\.conda\envs\skin_rehab\python.exe")
    else:
        print("⚠️ UNKNOWN: Could not automatically detect vendor.")
        
except Exception as e:
    print(f"❌ Error reading GPU info: {e}")

print("="*40)