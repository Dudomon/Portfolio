import 'package:flutter/foundation.dart';
import 'api_service.dart';
import '../models/podcast.dart';

class PodcastService extends ChangeNotifier {
  List<Podcast> _podcasts = [];
  bool _isLoading = false;
  String? _error;

  List<Podcast> get podcasts => _podcasts;
  List<Podcast> get latestPodcasts => _podcasts.take(3).toList();
  bool get isLoading => _isLoading;
  String? get error => _error;

  PodcastService() {
    loadPodcasts();
  }

  Future<void> loadPodcasts() async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      debugPrint('🎙️ Carregando podcasts do servidor');
      final response = await ApiService.get('/wp-content/uploads/podcasts/get_podcasts.php');
      
      if (response['status'] == 'success' && response['data'] != null) {
        _podcasts = (response['data'] as List)
            .map((json) => Podcast.fromJson(json))
            .toList();
        debugPrint('✅ ${_podcasts.length} podcasts carregados');
      } else {
        debugPrint('❌ Fallback para dados mockados');
        _podcasts = _getMockedPodcasts();
      }
    } catch (e) {
      _error = 'Erro ao carregar podcasts: $e';
      _podcasts = _getMockedPodcasts();
      debugPrint('💥 Erro ao carregar podcasts: $e');
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  Future<void> refreshPodcasts() async {
    await loadPodcasts();
  }

  // Se não conseguir carregar do servidor, retorna lista vazia
  List<Podcast> _getMockedPodcasts() {
    return []; // Lista vazia - sem dados falsos
  }
}