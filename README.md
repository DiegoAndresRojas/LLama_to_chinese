# LLama_to_chinese — configuração e execução local

Este repositório contém código para integrar um modelo LLaMA com restrições gramaticais (XGrammar) para gerar representações estruturais de caracteres chineses.

## Pré-requisitos

- Python 3.10+ (testado em 3.10 — 3.14). Recomenda-se usar uma versão estável do Python 3.10–3.14.
- Git
- (Opcional) GPU com CUDA adequada para execução acelerada do modelo. Sem GPU, a execução ocorrerá na CPU e pode ser lenta ou consumir muita memória dependendo do modelo.

## Passos para configurar o ambiente (Linux / macOS)

1. Criar o ambiente virtual na raiz do projeto:

   python3 -m venv .venv

2. Ativar o ambiente virtual:

   source .venv/bin/activate

3. Atualizar as ferramentas de instalação e instalar dependências:

   python -m pip install --upgrade pip setuptools wheel
   python -m pip install -r requirements.txt

Observação sobre o PyTorch: para obter o binário de PyTorch correto (por exemplo com suporte CUDA), prefira seguir as instruções oficiais em https://pytorch.org/get-started/locally/ e instalar a versão adequada ao seu sistema. Se não precisar de GPU, a instalação padrão via requirements.txt (ou `pip install torch`) costuma funcionar.

## Passos para Windows (PowerShell)

1. Criar o venv:

   python -m venv .venv

2. Ativar (PowerShell):

   .\.venv\Scripts\Activate.ps1

3. Atualizar pip e instalar dependências:

   python -m pip install --upgrade pip setuptools wheel
   python -m pip install -r requirements.txt

## Variáveis de ambiente úteis

- HF_TOKEN — token do Hugging Face (opcional, recomendado para evitar limites anônimos ao baixar modelos):

  export HF_TOKEN="seu_token_aqui"   # Linux/macOS
  setx HF_TOKEN "seu_token_aqui"     # Windows (PowerShell/Prompt)

Também é possível rodar `huggingface-cli login` para configurar credenciais.

## Dados (SQLite) e licenciamento

Por proteção de direitos autorais e distribuição, os arquivos SQLite usados pelo projeto NÃO são incluídos no repositório. Se o fluxo do projeto requer arquivos em `assets/data/*.sqlite`, obtenha-os junto às fontes autorizadas ou ao responsável pelo projeto e coloque-os localmente em `assets/data/`.

Para facilitar a reprodução, crie a pasta caso ela não exista e coloque os arquivos necessários:

   mkdir -p assets/data
   # copiar ou mover os .sqlite para assets/data/

## Como executar os exemplos

Com o venv ativado, execute os scripts diretamente com Python. Exemplos:

- Rodar a geração/compilação da gramática e testes:

  python src/app_ia/_com_spwc_bnf.py

- Scripts auxiliares (exemplos):

  python src/app_ia/_sem.py
  python src/app_ia/app_com.py

Observação: alguns scripts carregam modelos grandes e precisam de acesso à internet (Hugging Face) e recursos de memória. Para executar apenas partes que não exigem modelo, revise o código e comente o carregamento do modelo.

## Erros comuns e como resolver

- ModuleNotFoundError: No module named 'transformers'
  - Certifique-se de ativar o venv e instalar dependências: `source .venv/bin/activate` + `pip install -r requirements.txt`.

- Erros relacionados a XGrammar / Unicode ao compilar a gramática
  - Verifique a codificação dos arquivos de gramática (UTF-8) e a integridade dos arquivos em `assets/grammar`. O código inclui uma sanitização de gramática para tentar filtrar linhas inválidas, mas regras malformadas podem precisar de limpeza manual.

- Problemas de memória ou GPU
  - Use um modelo menor, rode no modo CPU (sem device_map) ou execute em uma máquina com mais memória/GPU.

## Recomendação para reprodução

1. Criar e ativar o venv
2. Instalar dependências
3. Colocar os arquivos SQLite em `assets/data/` (se necessários)
4. Rodar o script desejado com `python src/app_ia/_com_spwc_bnf.py`

## Ajuda adicional

Se ocorrerem erros durante a execução, copie o traceback e abra uma issue ou solicite suporte descrevendo:
- Comandos executados
- Versão do Python
- Sistemas operacionais
- Mensagens de erro completas

---

(Arquivo gerado/atualizado automaticamente para orientar execução local.)