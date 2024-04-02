import os
from PyQt5.QtCore import QLibraryInfo
# from PySide2.QtCore import QLibraryInfo
import cv2
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)

