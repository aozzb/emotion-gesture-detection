import cv2
import os
import numpy as np

class EmoteMapper:
    def __init__(self, emote_dir="../assets/emotes"):
        self.emotes={}
        self.load_emotes(emote_dir)

        self.size=(200, 200)
        self.window_name="EMOTE OUTPUT"

        self.current_emote=None

    def load_emotes(self,directory):
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Emote directory not found: {directory}")
        
        for file in os.listdir(directory):
            if file.lower().endswith(".png"):
                key=file.split(".")[0].lower()
                path=os.path.join(directory, file)

                img=cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue

                self.emotes[key]=img
        print("[EmoteMapper] Loaded emotes:", list(self.emotes.keys()))

    def resolve_emote(self, expression, gesture):
        #decides priority order of emotes
        if gesture and gesture in self.emotes:
            return gesture
        
        if expression and expression in self.emotes:
            return expression
        
        return None
    
    def render(self, emote_key):
        #show emote in a separate window

        canvas=np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)

        if emote_key is None or emote_key not in self.emotes:
            cv2.imshow(self.window_name, canvas)
            return
        
        emote=self.emotes[emote_key]
        emote=cv2.resize(emote, self.size, interpolation=cv2.INTER_AREA)
        
        if emote.shape[2]==4:
            alpha=emote[:,:, 3]/255.0
            for c in range(3):
                canvas[:, :, c]= (
                    alpha*emote[:, :, c]+
                    (1-alpha)*canvas[:, :, c]
                )
        else:
            canvas[:]=emote[:, :, :3]

        cv2.imshow(self.window_name, canvas)
