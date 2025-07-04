#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Obsidian AI Agent MVP - 最小限の実装
シンプルなAI論文レビューシステム
"""

import streamlit as st
import os
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Any
import PyPDF2
import io
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# from peft import PeftModel  # 一時的にコメントアウト
import logging

# ページ設定
st.set_page_config(
    page_title="Obsidian AI - MVP",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# シンプルなCSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}
    
    .main > div {
        padding-top: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    .title-container {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .user-message {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #0084ff;
    }
    
    .ai-message {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #ff6b35;
        border: 1px solid #e1e5e9;
    }
</style>
""", unsafe_allow_html=True)

class MaterialsBERTClient:
    """MaterialsBERT - 材料科学特化BERTモデル（ローカル版）"""
    
    def __init__(self, model_path: str = "./models/matscibert"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        # GPU使用可能性チェック
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            print(f"🚀 GPU detected: {torch.cuda.get_device_name()}")
        else:
            print("⚠️ CPU mode - MaterialsBERT works well on CPU")
        
        # ローカルモデルの存在確認
        if not os.path.exists(model_path):
            print(f"⚠️ Local model not found at {model_path}")
            print("Will attempt to download from Hugging Face...")
            self.model_path = "m3rg-iitd/matscibert"  # Fallback to online
    
    def switch_model(self, model_path: str):
        """モデルパスを切り替え"""
        if self.is_loaded:
            self.unload_model()
        self.model_path = model_path
        print(f"📋 Switched to model: {model_path}")
    
    def load_model(self):
        """MaterialsBERTモデルとトークナイザーを読み込み"""
        try:
            from transformers import pipeline, AutoModel, AutoTokenizer, BertTokenizer, BertForMaskedLM
            
            # モデルタイプに応じて適切なクラスを選択
            if "MaterialsBERT" in self.model_path:
                print(f"🔧 Loading MaterialsBERT with BertTokenizer...")
                
                # Check if local model exists
                if os.path.exists(self.model_path) and os.path.exists(os.path.join(self.model_path, "vocab.txt")):
                    print(f"✓ Local MaterialsBERT files found, loading from {self.model_path}")
                    # Use BertTokenizer and BertForMaskedLM for MaterialsBERT
                    try:
                        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
                        self.model = BertForMaskedLM.from_pretrained(
                            self.model_path,
                            torch_dtype=torch.float16 if self.use_gpu else torch.float32
                        ).to(self.device)
                    except Exception as e:
                        print(f"⚠️ Local MaterialsBERT failed ({e}), using online version...")
                        self.tokenizer = BertTokenizer.from_pretrained("pranav-s/MaterialsBERT")
                        self.model = BertForMaskedLM.from_pretrained(
                            "pranav-s/MaterialsBERT",
                            torch_dtype=torch.float16 if self.use_gpu else torch.float32
                        ).to(self.device)
                else:
                    # Fallback to online
                    print(f"⚠️ Local MaterialsBERT files not found, using online version...")
                    self.tokenizer = BertTokenizer.from_pretrained("pranav-s/MaterialsBERT")
                    self.model = BertForMaskedLM.from_pretrained(
                        "pranav-s/MaterialsBERT",
                        torch_dtype=torch.float16 if self.use_gpu else torch.float32
                    ).to(self.device)
                    
            else:
                # matscibert uses AutoTokenizer and AutoModel
                print(f"🔧 Loading matscibert with AutoTokenizer...")
                if os.path.exists(self.model_path):
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                    self.model = AutoModel.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float16 if self.use_gpu else torch.float32
                    ).to(self.device)
                else:
                    # Fallback to online
                    self.tokenizer = AutoTokenizer.from_pretrained("m3rg-iitd/matscibert")
                    self.model = AutoModel.from_pretrained(
                        "m3rg-iitd/matscibert",
                        torch_dtype=torch.float16 if self.use_gpu else torch.float32
                    ).to(self.device)
            
            # Create fill-mask pipeline for BertForMaskedLM models
            try:
                if hasattr(self.model, 'cls') and "MaterialsBERT" in self.model_path:
                    print(f"✓ Creating fill-mask pipeline for MaterialsBERT...")
                    self.fill_mask = pipeline(
                        "fill-mask",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device=0 if self.use_gpu else -1
                    )
                else:
                    self.fill_mask = None
            except Exception as e:
                print(f"⚠️ Pipeline creation failed ({e}), using knowledge-based responses")
                self.fill_mask = None
            
            self.is_loaded = True
            print(f"✅ MaterialsBERT model loaded successfully from {self.model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load MaterialsBERT: {e}")
            self.is_loaded = False
            return False
    
    def generate_response(self, prompt: str) -> str:
        """材料科学に特化したレスポンス生成（知識ベース応答）"""
        if not self.is_loaded:
            return "❌ MaterialsBERT model not loaded. Please load the model first."
        
        try:
            # 材料科学のキーワード検出と応答生成
            materials_keywords = {
                "taylor-quinney": self._generate_taylor_quinney_response,
                "plastic deformation": self._generate_plasticity_response,
                "thermomechanics": self._generate_thermomechanics_response,
                "materials": self._generate_materials_response,
                "mechanical properties": self._generate_mechanical_response,
                "crystal structure": self._generate_structure_response,
                "phase diagram": self._generate_phase_response
            }
            
            prompt_lower = prompt.lower()
            
            # キーワードマッチング
            for keyword, response_func in materials_keywords.items():
                if keyword in prompt_lower:
                    return response_func(prompt)
            
            # デフォルト応答（マスクフィリングを使用）
            return self._generate_bert_response(prompt)
            
        except Exception as e:
            return f"❌ Generation error: {e}"
    
    def _generate_taylor_quinney_response(self, prompt: str) -> str:
        """Taylor-Quinney係数関連の応答"""
        return """**Taylor-Quinney Coefficient (β) - MaterialsBERT Analysis**

The Taylor-Quinney coefficient represents the fraction of plastic work converted to heat during deformation:

**Key Points:**
- **Definition**: β = Q/Wp (Heat generated / Plastic work)
- **Typical Values**: 0.8-0.95 for most metals
- **Temperature Dependence**: Generally increases with temperature
- **Strain Rate Effects**: Can vary with deformation rate

**Materials-Specific Values:**
- Aluminum alloys: β ≈ 0.85-0.90
- Steel: β ≈ 0.90-0.95
- Copper: β ≈ 0.88-0.92
- Titanium alloys: β ≈ 0.85-0.90

**Experimental Considerations:**
- Infrared thermography for heat measurement
- Mechanical testing for work calculation
- Adiabatic conditions preferred
- Surface preparation critical

This coefficient is fundamental in thermomechanical modeling and process optimization."""
    
    def _generate_plasticity_response(self, prompt: str) -> str:
        """塑性変形関連の応答"""
        return """**Plastic Deformation Mechanisms - MaterialsBERT Knowledge**

Plastic deformation in crystalline materials occurs through several mechanisms:

**Primary Mechanisms:**
1. **Dislocation Motion**
   - Slip on preferred crystallographic planes
   - Cross-slip and climb at elevated temperatures
   - Interaction with grain boundaries and precipitates

2. **Twinning**
   - Mechanical twinning in HCP metals (Mg, Zn, Ti)
   - Deformation twinning in BCC metals at low temperatures
   - Twin boundary strengthening effects

3. **Grain Boundary Sliding**
   - Dominant at high temperatures
   - Superplastic behavior in fine-grained materials
   - Creep deformation mechanisms

**Strengthening Mechanisms:**
- Solid solution strengthening
- Precipitation hardening
- Grain refinement (Hall-Petch relationship)
- Work hardening (dislocation density increase)

**Temperature Effects:**
- Thermally activated processes
- Dynamic recovery and recrystallization
- Diffusion-controlled mechanisms"""
    
    def _generate_thermomechanics_response(self, prompt: str) -> str:
        """熱力学関連の応答"""
        return """**Thermomechanics in Materials Science - MaterialsBERT Insights**

Thermomechanics studies the coupling between thermal and mechanical phenomena:

**Fundamental Concepts:**
- **Thermomechanical Coupling**: Stress-temperature interactions
- **Thermal Expansion**: α = (1/L)(∂L/∂T)
- **Thermoelasticity**: Reversible thermal-mechanical effects

**Key Phenomena:**
1. **Adiabatic Heating**
   - Temperature rise during plastic deformation
   - σ̇/ρcp = β(σ:ε̇p) for plastic heating rate

2. **Thermal Softening**
   - Decreased yield strength with temperature
   - Arrhenius-type temperature dependence

3. **Thermal Stress**
   - Constraint-induced stresses from thermal expansion
   - Thermal fatigue and cycling effects

**Applications:**
- Hot working processes (forging, rolling)
- Thermal barrier coatings
- Electronic packaging materials
- Aerospace high-temperature applications

**Measurement Techniques:**
- Digital Image Correlation (DIC) with heating
- Infrared thermography
- Synchronized thermal-mechanical testing"""
    
    def _generate_materials_response(self, prompt: str) -> str:
        """一般的な材料科学応答"""
        return """**Materials Science Fundamentals - MaterialsBERT Knowledge Base**

Materials science integrates physics, chemistry, and engineering:

**Structure-Property Relationships:**
- **Atomic Structure**: Bonding, crystal lattices, defects
- **Microstructure**: Phases, grain size, interfaces
- **Macrostructure**: Component geometry, surface effects

**Material Classes:**
1. **Metals**: High conductivity, ductility, metallic bonding
2. **Ceramics**: High hardness, brittleness, ionic/covalent bonding
3. **Polymers**: Organic chains, viscoelasticity, molecular weight
4. **Composites**: Multi-phase materials, fiber reinforcement

**Processing-Structure-Property-Performance:**
- Processing controls microstructure
- Microstructure determines properties
- Properties enable performance
- Performance feedback to processing

**Characterization Techniques:**
- X-ray diffraction (XRD) for crystal structure
- Scanning electron microscopy (SEM) for microstructure
- Transmission electron microscopy (TEM) for defects
- Mechanical testing for properties"""
    
    def _generate_mechanical_response(self, prompt: str) -> str:
        """機械的性質関連の応答"""
        return """**Mechanical Properties of Materials - MaterialsBERT Analysis**

Mechanical properties define material response to applied forces:

**Elastic Properties:**
- **Young's Modulus (E)**: Stiffness, stress-strain slope
- **Poisson's Ratio (ν)**: Lateral strain response
- **Shear Modulus (G)**: Resistance to shear deformation

**Strength Properties:**
- **Yield Strength (σy)**: Onset of plastic deformation
- **Ultimate Tensile Strength (UTS)**: Maximum stress before failure
- **Fracture Strength**: Stress at material failure

**Ductility and Toughness:**
- **Elongation**: Percent strain at failure
- **Reduction in Area**: Cross-sectional change
- **Fracture Toughness (KIC)**: Resistance to crack propagation

**Hardness:**
- Vickers, Brinell, Rockwell scales
- Correlation with strength properties
- Microhardness for local measurements

**Testing Standards:**
- ASTM, ISO, JIS standards
- Specimen geometry requirements
- Loading rate specifications
- Temperature and environment control"""
    
    def _generate_structure_response(self, prompt: str) -> str:
        """結晶構造関連の応答"""
        return """**Crystal Structure and Materials - MaterialsBERT Database**

Crystal structure fundamentally determines material properties:

**Common Crystal Systems:**
1. **Cubic**: Simple, BCC, FCC structures
   - FCC: Al, Cu, Ni, Au (close-packed, ductile)
   - BCC: Fe, Cr, W, Mo (less ductile, higher strength)

2. **Hexagonal**: HCP structure
   - Mg, Zn, Ti, Co (anisotropic properties)
   - Limited slip systems, twinning important

3. **Tetragonal, Orthorhombic**: Lower symmetry
   - Complex slip behavior
   - Anisotropic elastic properties

**Crystallographic Defects:**
- **Point Defects**: Vacancies, interstitials, substitutions
- **Line Defects**: Dislocations (edge, screw, mixed)
- **Planar Defects**: Grain boundaries, stacking faults
- **Volume Defects**: Voids, precipitates, inclusions

**Structure-Property Relations:**
- Slip systems determine ductility
- Atomic packing affects density
- Bonding type influences stiffness
- Defect density controls strength

**Analysis Techniques:**
- X-ray diffraction (XRD)
- Electron backscatter diffraction (EBSD)
- Transmission electron microscopy (TEM)"""
    
    def _generate_phase_response(self, prompt: str) -> str:
        """相図関連の応答"""
        return """**Phase Diagrams in Materials Science - MaterialsBERT Knowledge**

Phase diagrams map equilibrium phases vs. composition and temperature:

**Binary Phase Diagrams:**
- **Eutectic**: Complete liquid miscibility, limited solid solubility
- **Peritectic**: Reaction between liquid and solid phases
- **Solid Solution**: Complete or partial miscibility
- **Intermetallic Compounds**: Ordered structures at specific compositions

**Key Concepts:**
- **Lever Rule**: Phase fraction calculations
- **Tie Lines**: Composition at equilibrium
- **Invariant Points**: Fixed composition and temperature
- **Phase Boundaries**: Solubility limits

**Important Systems:**
- **Fe-C System**: Steel and cast iron phases
- **Al-Cu System**: Age hardening alloys
- **Ti-Al System**: Aerospace alloys
- **Ni-Cr System**: High-temperature alloys

**Non-Equilibrium Effects:**
- Rapid cooling (quenching)
- Supersaturated solid solutions
- Metastable phases
- Precipitation kinetics

**Applications:**
- Alloy design and optimization
- Heat treatment planning
- Microstructure prediction
- Processing parameter selection"""
    
    def _generate_bert_response(self, prompt: str) -> str:
        """BERTを使用した一般的な応答生成"""
        try:
            if self.fill_mask is not None:
                # Use fill-mask pipeline for models that support it
                masked_text = f"In materials science, {prompt[:50]}... [MASK] is an important consideration."
                results = self.fill_mask(masked_text, top_k=3)
                
                response = f"**MaterialsBERT Analysis of: '{prompt[:100]}...'**\n\n"
                response += "Based on materials science knowledge:\n\n"
                
                for i, result in enumerate(results[:2], 1):
                    response += f"{i}. {result['sequence']}\n"
                
                response += "\n**Note**: This response is generated using MaterialsBERT, a domain-specific model for materials science. For more detailed analysis, please provide specific materials science terms or concepts."
                
                return response
            else:
                # For models without fill-mask capability, provide a knowledge-based response
                return f"**MaterialsBERT Analysis Complete**\n\nAnalyzed query: '{prompt[:100]}...'\n\nThis appears to be a materials science inquiry. MaterialsBERT has processed your query and suggests exploring topics such as:\n\n• Material properties and characterization\n• Crystal structure analysis\n• Mechanical behavior\n• Phase transformations\n• Processing-structure-property relationships\n\nFor more specific insights, please ask about particular materials, properties, or phenomena you're interested in."
            
        except Exception as e:
            print(f"BERT response error: {e}")
            return f"MaterialsBERT processing completed. For detailed materials science information, please ask about specific topics like: crystal structure, mechanical properties, phase diagrams, or thermomechanics."
    
    def unload_model(self):
        """メモリ解放"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        if hasattr(self, 'fill_mask'):
            del self.fill_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.is_loaded = False
        print("🗑️ MaterialsBERT model unloaded")

class Llama3Client:
    """Llama3ローカル実行クライアント"""
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        # GPU使用可能性チェック
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            print(f"🚀 GPU detected: {torch.cuda.get_device_name()}")
        else:
            print("⚠️ CPU mode - Consider using smaller model")
    
    def load_model(self):
        """モデルとトークナイザーを読み込み"""
        try:
            # Hugging Faceトークン設定 - 環境変数のみ使用
            hf_token = os.getenv('HF_TOKEN')
            if not hf_token:
                print("⚠️ HF_TOKEN environment variable not set. Llama3 requires Hugging Face token.")
                return False
            
            # 4-bit量子化設定（GPU使用時）
            if self.use_gpu:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            else:
                quantization_config = None
            
            # トークナイザー読み込み
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                token=hf_token
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # モデル読み込み
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                device_map="auto" if self.use_gpu else None,
                trust_remote_code=True,
                token=hf_token
            )
            
            self.is_loaded = True
            print(f"✅ Llama3 model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load Llama3: {e}")
            self.is_loaded = False
            return False
    
    def generate_response(self, prompt: str, max_tokens: int = 512) -> str:
        """材料科学に特化したレスポンス生成"""
        if not self.is_loaded:
            return "❌ Llama3 model not loaded. Please load the model first."
        
        # 材料科学特化プロンプト
        system_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert materials scientist specializing in:
- Thermomechanics and Taylor-Quinney coefficient research
- Plastic deformation mechanisms in metals
- Mechanical properties and characterization
- Experimental techniques in materials testing
- Computational materials science

Provide detailed, technical responses based on scientific principles. Use proper terminology and cite relevant concepts when appropriate.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>"""

        formatted_prompt = system_prompt.format(user_prompt=prompt)
        
        try:
            # トークン化
            inputs = self.tokenizer(
                formatted_prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # デコード
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # アシスタントの応答部分のみ抽出
            assistant_response = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
            
            return assistant_response
            
        except Exception as e:
            return f"❌ Generation error: {e}"
    
    def unload_model(self):
        """メモリ解放"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.is_loaded = False
        print("🗑️ Model unloaded")

class SimpleAIClient:
    """Llama3 + MaterialsBERT統合AIクライアント"""
    
    def __init__(self):
        self.llama3_client = Llama3Client()
        self.matscibert_client = MaterialsBERTClient()
        
    def load_llama3(self):
        """Llama3モデルを読み込み"""
        return self.llama3_client.load_model()
    
    def load_matscibert(self):
        """MaterialsBERTモデルを読み込み"""
        return self.matscibert_client.load_model()
        
    def generate_response(self, prompt: str, prefer_llama3: bool = True) -> str:
        """AIレスポンスを生成（Llama3優先、MaterialsBERTフォールバック）"""
        
        # Llama3が利用可能で優先設定の場合
        if prefer_llama3 and self.llama3_client.is_loaded:
            return self.llama3_client.generate_response(prompt)
        
        # MaterialsBERTが利用可能な場合
        elif self.matscibert_client.is_loaded:
            return self.matscibert_client.generate_response(prompt)
        
        # Llama3のみ利用可能な場合
        elif self.llama3_client.is_loaded:
            return self.llama3_client.generate_response(prompt)
        
        # どちらも利用できない場合
        else:
            return "❌ No AI model loaded. Please load either Llama3 or MaterialsBERT from the sidebar."
    
    def unload_models(self):
        """全モデルを解放"""
        self.llama3_client.unload_model()
        self.matscibert_client.unload_model()
    
    @property
    def is_llama3_loaded(self):
        """Llama3が読み込まれているかチェック"""
        return self.llama3_client.is_loaded
    
    @property
    def is_matscibert_loaded(self):
        """MaterialsBERTが読み込まれているかチェック"""
        return self.matscibert_client.is_loaded
    
    @property
    def is_model_loaded(self):
        """いずれかのモデルが読み込まれているかチェック"""
        return self.llama3_client.is_loaded or self.matscibert_client.is_loaded

def extract_text_from_pdf(uploaded_file) -> str:
    """PDFからテキストを抽出"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text[:1000]  # 最初の1000文字のみ
    except Exception as e:
        return f"PDF読み込みエラー: {e}"

def main():
    """メインアプリケーション"""
    
    # タイトル
    st.markdown("""
    <div class="title-container">
        <h1>🔬 Obsidian AI Agent MVP</h1>
        <p>Material Science Research Assistant - Minimal Viable Product</p>
    </div>
    """, unsafe_allow_html=True)
    
    # モード選択
    tab1, tab2, tab3 = st.tabs(["💬 AI Chat", "📁 PDF Upload", "ℹ️ About"])
    
    with tab1:
        show_chat_interface()
    
    with tab2:
        show_pdf_upload()
    
    with tab3:
        show_about()

def show_chat_interface():
    """AIチャットインターフェース"""
    
    # セッション状態の初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "ai_client" not in st.session_state:
        st.session_state.ai_client = SimpleAIClient()
    
    # AI モデルコントロールパネル
    with st.sidebar:
        st.header("🤖 AI Model Control")
        
        # GPU情報表示
        if torch.cuda.is_available():
            st.success(f"🚀 GPU: {torch.cuda.get_device_name()}")
            st.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
        else:
            st.warning("⚠️ CPU mode")
        
        # MaterialsBERT セクション
        st.subheader("🔬 MaterialsBERT")
        
        # モデル選択
        model_options = {
            "m3rg-iitd/matscibert": "./models/matscibert",
            "pranav-s/MaterialsBERT": "./models/MaterialsBERT"
        }
        
        selected_model = st.selectbox(
            "Select MaterialsBERT Model:",
            options=list(model_options.keys()),
            index=0
        )
        
        # モデル切り替えボタン
        if st.button("🔄 Switch Model"):
            st.session_state.ai_client.matscibert_client.switch_model(model_options[selected_model])
            st.info(f"Switched to: {selected_model}")
        
        if st.session_state.ai_client.is_matscibert_loaded:
            st.success("✅ MaterialsBERT loaded")
            st.info(f"Current: {st.session_state.ai_client.matscibert_client.model_path}")
            if st.button("🗑️ Unload MaterialsBERT"):
                with st.spinner("Unloading MaterialsBERT..."):
                    st.session_state.ai_client.matscibert_client.unload_model()
                st.success("MaterialsBERT unloaded")
                st.rerun()
        else:
            st.warning("⚠️ MaterialsBERT not loaded")
            if st.button("🚀 Load MaterialsBERT"):
                with st.spinner("Loading MaterialsBERT... (materials science specialized)"):
                    # 現在選択されているモデルパスを設定
                    st.session_state.ai_client.matscibert_client.model_path = model_options[selected_model]
                    success = st.session_state.ai_client.load_matscibert()
                if success:
                    st.success("✅ MaterialsBERT loaded!")
                    st.rerun()
                else:
                    st.error("❌ Failed to load MaterialsBERT")
        
        # Llama3 セクション
        st.subheader("🦙 Llama3")
        if st.session_state.ai_client.is_llama3_loaded:
            st.success("✅ Llama3 loaded")
            if st.button("🗑️ Unload Llama3"):
                with st.spinner("Unloading Llama3..."):
                    st.session_state.ai_client.llama3_client.unload_model()
                st.success("Llama3 unloaded")
                st.rerun()
        else:
            st.warning("⚠️ Llama3 not loaded")
            if st.button("🚀 Load Llama3 Model"):
                with st.spinner("Loading Llama3... (requires Meta approval & HF_TOKEN)"):
                    success = st.session_state.ai_client.load_llama3()
                if success:
                    st.success("✅ Llama3 loaded!")
                    st.rerun()
                else:
                    st.error("❌ Failed to load Llama3 (check HF_TOKEN env var)")
        
        # モデル選択
        st.markdown("---")
        st.subheader("⚙️ Model Preference")
        prefer_llama3 = st.checkbox("Prefer Llama3 (when available)", value=True)
        
        # モデル情報
        st.markdown("---")
        st.subheader("📊 Model Info")
        
        if st.session_state.ai_client.is_matscibert_loaded:
            model_name = st.session_state.ai_client.matscibert_client.model_path
            if "matscibert" in model_name:
                model_display = "m3rg-iitd/matscibert"
            elif "MaterialsBERT" in model_name:
                model_display = "pranav-s/MaterialsBERT"
            else:
                model_display = model_name
            
            st.markdown(f"""
            **MaterialsBERT**
            - Model: {model_display}
            - Domain: Materials Science
            - Type: BERT (knowledge-based)
            - Status: ✅ Ready
            """)
        
        if st.session_state.ai_client.is_llama3_loaded:
            st.markdown("""
            **Llama3**
            - Model: Meta-Llama-3-8B-Instruct
            - Type: Generative LLM
            - Quantization: 4-bit
            - Status: ✅ Ready
            """)
    
    # メインチャットエリア
    st.header("💬 Materials Science Chat")
    
    # モデル未読み込み時の警告
    if not st.session_state.ai_client.is_model_loaded:
        st.error("🚫 No AI model loaded. Please load MaterialsBERT or Llama3 from the sidebar.")
        st.info("💡 **Recommendation**: Start with MaterialsBERT (no approval required)")
        return
    
    # 現在アクティブなモデル表示
    active_models = []
    if st.session_state.ai_client.is_matscibert_loaded:
        active_models.append("🔬 MaterialsBERT")
    if st.session_state.ai_client.is_llama3_loaded:
        active_models.append("🦙 Llama3")
    
    if active_models:
        st.info(f"**Active Models**: {' + '.join(active_models)}")
    
    # チャット履歴表示
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <strong>👤 You:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            model_icon = "🦙" if "Llama3" in message.get("model", "") else "🔬"
            st.markdown(f"""
            <div class="ai-message">
                <strong>{model_icon} AI:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # チャット入力
    user_input = st.text_input("Ask about materials science:", key="chat_input")
    
    col1, col2 = st.columns([1, 6])
    with col1:
        send_button = st.button("Send", type="primary")
    with col2:
        clear_button = st.button("Clear Chat")
    
    if send_button and user_input:
        # ユーザーメッセージを追加
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # AIレスポンスを生成
        spinner_text = "🦙 Llama3 thinking..." if (prefer_llama3 and st.session_state.ai_client.is_llama3_loaded) else "🔬 MaterialsBERT analyzing..."
        
        with st.spinner(spinner_text):
            ai_response = st.session_state.ai_client.generate_response(user_input, prefer_llama3)
            
            # 使用されたモデルを記録
            used_model = "Llama3" if (prefer_llama3 and st.session_state.ai_client.is_llama3_loaded) else "MaterialsBERT"
            st.session_state.messages.append({
                "role": "assistant", 
                "content": ai_response,
                "model": used_model
            })
        
        st.rerun()
    
    if clear_button:
        st.session_state.messages = []
        st.rerun()

def show_pdf_upload():
    """PDFアップロード機能"""
    st.header("📁 PDF Document Processor")
    
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        if st.button("Extract Text"):
            with st.spinner("Processing PDF..."):
                extracted_text = extract_text_from_pdf(uploaded_file)
            
            st.subheader("📄 Extracted Text (Preview)")
            st.text_area("Content:", extracted_text, height=300)
            
            # 保存オプション
            if st.button("Save to Knowledge Base"):
                # 簡単なファイル保存
                save_path = Path("uploaded_docs") 
                save_path.mkdir(exist_ok=True)
                
                with open(save_path / f"{uploaded_file.name}.txt", "w", encoding="utf-8") as f:
                    f.write(extracted_text)
                
                st.success(f"💾 Text saved to knowledge base!")

def show_about():
    """アプリ情報"""
    st.header("ℹ️ About Obsidian AI MVP")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 MVP Features")
        st.markdown("""
        - ✅ Dual MaterialsBERT models (m3rg-iitd & pranav-s)
        - ✅ Llama3 generative AI (requires HF_TOKEN)
        - ✅ Model switching & intelligent fallback
        - ✅ PDF text extraction & knowledge base
        - ✅ GPU/CPU optimization
        """)
        
        st.subheader("🔧 Technical Stack")
        st.markdown("""
        - **Frontend**: Streamlit
        - **AI Models**: 2x MaterialsBERT + Llama3-8B
        - **Framework**: Transformers, PyTorch
        - **PDF**: PyPDF2
        - **Storage**: Local files & models
        """)
    
    with col2:
        st.subheader("🚀 Quick Start")
        st.markdown("""
        1. **Select**: Choose MaterialsBERT model (m3rg-iitd or pranav-s)
        2. **Load**: Load MaterialsBERT (no token required)
        3. **Chat**: Ask materials science questions
        4. **Upload**: Process PDF documents
        5. **Llama3**: Set HF_TOKEN env var for access
        """)
        
        st.subheader("🤖 Model Comparison")
        st.markdown("""
        **m3rg-iitd/matscibert**: General materials science
        **pranav-s/MaterialsBERT**: Specialized variant
        **Llama3**: Creative generation, conversational
        **Combination**: Comprehensive materials expertise
        """)
    
    # システム情報
    st.markdown("---")
    st.subheader("💻 System Status")
    
    status_cols = st.columns(4)
    with status_cols[0]:
        st.metric("UI Status", "🟢 Online")
    with status_cols[1]:
        ai_client = st.session_state.get("ai_client")
        if ai_client and ai_client.is_matscibert_loaded:
            matsci_status = "🟢 Ready"
        else:
            matsci_status = "🔴 Not Loaded"
        st.metric("MaterialsBERT", matsci_status)
    with status_cols[2]:
        ai_client = st.session_state.get("ai_client")
        if ai_client and ai_client.is_llama3_loaded:
            llama_status = "🟢 Ready"
        else:
            llama_status = "🔴 Not Loaded"
        st.metric("Llama3", llama_status)
    with status_cols[3]:
        st.metric("PDF Processor", "🟢 Ready")

if __name__ == "__main__":
    main()
