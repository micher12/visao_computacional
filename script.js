// Esta função é chamada pelo Python para atualizar o feed de vídeo.
function updateVideoFeed(base64Image) {
    const videoFeed = document.getElementById('video-feed');
    videoFeed.src = 'data:image/jpeg;base64,' + base64Image;
}


document.addEventListener('DOMContentLoaded', () => {
    const extractBtn = document.getElementById('extract-btn');
    const resultsDiv = document.getElementById('results');

    // Adiciona um evento de clique ao botão.
    extractBtn.addEventListener('click', async () => {
        // Desabilita o botão e mostra uma mensagem de carregamento.
        extractBtn.disabled = true;
        extractBtn.innerText = 'Analisando...';
        resultsDiv.innerText = 'Processando a imagem com a IA. Por favor, aguarde...';

        try {
            // Chama a função 'extract_data' do backend Python.
            const report = await window.pywebview.api.extract_data();
            
            // Exibe o relatório retornado pelo Python.
            resultsDiv.innerText = report || 'Nenhum dado retornado.';

        } catch (error) {
            // Em caso de erro, exibe uma mensagem de falha.
            console.error('Erro ao extrair dados:', error);
            resultsDiv.innerText = 'Ocorreu um erro ao tentar se comunicar com a API. Verifique o console para mais detalhes.';
        } finally {
            // Reabilita o botão, independentemente do resultado.
            extractBtn.disabled = false;
            extractBtn.innerText = 'Extrair Dados dos Materiais';
        }
    });
});