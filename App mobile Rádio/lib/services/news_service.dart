import 'package:flutter/foundation.dart';
import 'api_service.dart';
import '../models/news.dart';

class NewsService extends ChangeNotifier {
  List<News> _news = [];
  bool _isLoading = false;
  String? _error;

  List<News> get news => _news;
  bool get isLoading => _isLoading;
  String? get error => _error;

  NewsService() {
    loadNews();
  }

  Future<void> loadNews({int limit = 10}) async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      debugPrint('🔄 Carregando notícias de: /wp-content/noticias/get_noticias.php?limit=$limit');
      final response = await ApiService.get('/wp-content/noticias/get_noticias.php?limit=$limit');
      
      debugPrint('📡 Resposta da API: $response');
      
      if (response['status'] == 'success' && response['data'] != null) {
        final dataList = response['data'] as List;
        debugPrint('🔍 Primeira notícia raw: ${dataList.isNotEmpty ? dataList[0] : "vazia"}');
        
        final allNews = dataList
            .map((json) {
              debugPrint('📰 Processando notícia: ${json['titulo']} - Fonte: ${json['fonte']}');
              return News.fromJson(json);
            })
            .toList();
        
        // Filtrar notícias SEM imagem e duplicatas (igual ao widget)
        final Map<String, bool> seenIds = {};
        _news = allNews.where((noticia) {
          // Remove duplicatas por ID
          if (seenIds.containsKey(noticia.id)) {
            debugPrint('🚫 Duplicata removida: ${noticia.titulo}');
            return false;
          }
          seenIds[noticia.id] = true;
          
          // Remove notícias sem imagem
          if (noticia.imagem == null || noticia.imagem!.isEmpty) {
            debugPrint('🖼️ Notícia sem imagem removida: ${noticia.titulo}');
            return false;
          }
          
          return true;
        }).toList();
        
        debugPrint('✅ ${_news.length} notícias carregadas (${allNews.length - _news.length} filtradas)');
      } else {
        _error = 'Nenhuma notícia encontrada - Resposta: $response';
        debugPrint('❌ $_error');
      }
    } catch (e) {
      _error = 'Erro ao carregar notícias: $e';
      debugPrint('💥 $_error');
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  Future<void> refreshNews() async {
    await loadNews();
  }

  // Carrega mais notícias (paginação)
  Future<void> loadMoreNews() async {
    if (_isLoading) return;
    
    _isLoading = true;
    notifyListeners();

    try {
      final response = await ApiService.get(
        '/wp-content/noticias/get_noticias.php?limit=10&offset=${_news.length}'
      );
      
      if (response['status'] == 'success' && response['data'] != null) {
        final moreNews = (response['data'] as List)
            .map((json) => News.fromJson(json))
            .toList();
        
        // Filtrar duplicatas antes de adicionar
        final existingIds = _news.map((n) => n.id).toSet();
        final filteredNews = moreNews.where((noticia) {
          // Remove duplicatas por ID
          if (existingIds.contains(noticia.id)) {
            debugPrint('🚫 Duplicata removida no loadMore: ${noticia.titulo}');
            return false;
          }
          
          // Remove notícias sem imagem
          if (noticia.imagem == null || noticia.imagem!.isEmpty) {
            debugPrint('🖼️ Notícia sem imagem removida no loadMore: ${noticia.titulo}');
            return false;
          }
          
          return true;
        }).toList();
        
        _news.addAll(filteredNews);
        debugPrint('✅ ${filteredNews.length} novas notícias adicionadas (${moreNews.length - filteredNews.length} filtradas)');
      }
    } catch (e) {
      debugPrint('Erro ao carregar mais notícias: $e');
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }
}