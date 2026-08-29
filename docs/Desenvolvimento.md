# Relatório de Teste Inicial — Decodificação Restrita por Gramática com XGrammar

## 1. Introdução

Este relatório apresenta uma prova inicial de integração entre um modelo de linguagem da família LLaMA e o **XGrammar**, com o objetivo de verificar o funcionamento da decodificação restrita por gramática.

A proposta consiste em limitar a geração do modelo por meio de regras formais, sem modificar seus pesos ou realizar um novo processo de treinamento. No contexto do projeto, essas regras deverão posteriormente representar a gramática SWPC-BNF.

Nesta etapa, foi utilizada uma gramática reduzida apenas para validar o funcionamento básico da abordagem.

## 2. Preparação do ambiente

Inicialmente, foi criado um ambiente virtual em Python para isolar as bibliotecas utilizadas no experimento.

Em seguida, foram instaladas as principais dependências necessárias para executar o modelo e aplicar as restrições gramaticais, entre elas:

- PyTorch;
- Transformers;
- Hugging Face Hub;
- XGrammar.

Essa configuração permitiu executar localmente um modelo de linguagem e controlar sua saída durante a inferência.

Nota sobre os dados: os arquivos SQLite utilizados pelo projeto (por exemplo em assets/data/*.sqlite) não são incluídos no repositório por questões de proteção de direitos autorais e distribuição. Para reproduzir os experimentos, obtenha os arquivos necessários junto às fontes autorizadas ou ao responsável pelo projeto e coloque-os localmente na pasta assets/data. Certifique-se de ter permissão para usar e distribuir esses dados antes de adicioná-los ao repositório.

## 3. Seleção do modelo

O modelo inicialmente previsto para o teste foi:

```text
meta-llama/Llama-3.2-1B-Instruct
```

Esse modelo está disponível no Hugging Face, porém seu acesso é controlado pela Meta. Para utilizá-lo, é necessário autenticar uma conta no Hugging Face, aceitar os termos correspondentes e solicitar autorização para acessar o repositório.

Como a solicitação permanecia pendente de aprovação, decidiu-se utilizar temporariamente um modelo alternativo que não exigisse esse processo:

```text
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Essa substituição permitiu continuar a validação da metodologia sem alterar o objetivo principal do experimento.

## 4. Definição da gramática

Foi criada uma gramática EBNF simplificada com algumas estruturas e componentes básicos de caracteres chineses.

Um exemplo das regras utilizadas é:

```ebnf
root ::= terminal | lr | tb

lr ::= "LR(" terminal "," terminal ")"
tb ::= "TB(" terminal "," terminal ")"

terminal ::= "水" | "木" | "目" | "雨" | "日" | "口"
```

Essa gramática determina quais formatos podem ser produzidos pelo modelo.

No projeto completo, a mesma ideia deverá ser aplicada às regras SWPC-BNF, cuja tradução para um formato compatível com XGrammar faz parte da metodologia proposta.

## 5. Metodologia

O experimento foi realizado em duas condições.

Na primeira, o modelo gerou a resposta livremente, sem qualquer restrição sobre sua saída.

Na segunda, o mesmo modelo e o mesmo comando foram utilizados, porém com o XGrammar atuando durante a geração. Nesse caso, apenas sequências compatíveis com a gramática definida podiam ser produzidas.

O comando utilizado foi:

```text
Representa a estrutura do caractere chinês para água.
```

Essa comparação segue a lógica proposta no plano experimental, que prevê confrontar a geração convencional com a geração controlada por gramática.

## 6. Resultados

Na execução **sem XGrammar**, o modelo produziu uma saída livre que não correspondia ao formato estrutural esperado.

O resultado obtido foi:

```text
3. 0x00000000 - Representa a estrutura do caractere chinês para água.

4. 0x00000001 -
```

Na execução **com XGrammar**, o modelo produziu:

```text
TB(水,水)
```

A segunda resposta respeita a estrutura definida pela gramática, enquanto a primeira não apresenta uma representação compatível com esse formato.

É importante observar que o resultado com XGrammar demonstra apenas que a saída é **sintaticamente válida**. Isso não significa que `TB(水,水)` seja necessariamente a descrição estrutural correta do caractere `水`.

## 7. Conclusão

A prova inicial mostrou que o XGrammar pode ser integrado a um modelo baseado na arquitetura LLaMA para restringir sua saída durante a inferência.

Sem a restrição, o modelo produziu texto fora do formato esperado. Com a gramática ativa, a geração permaneceu dentro da estrutura definida.

O teste também evidencia uma distinção importante: a gramática pode controlar a **validade sintática**, mas não garante por si só a **correção semântica** da resposta.

Como próximo passo, a gramática simplificada deverá ser substituída progressivamente pelas regras SWPC-BNF utilizadas no projeto. A partir disso, será possível realizar uma comparação mais ampla entre geração livre e geração restrita, conforme previsto no plano de pesquisa.