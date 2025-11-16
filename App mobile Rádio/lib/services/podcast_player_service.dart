import 'package:flutter/foundation.dart';
import 'package:just_audio/just_audio.dart';
import '../models/podcast.dart';

class PodcastPlayerService extends ChangeNotifier {
  final AudioPlayer _audioPlayer = AudioPlayer();
  
  Podcast? _currentPodcast;
  bool _isPlaying = false;
  bool _isLoading = false;
  Duration _duration = Duration.zero;
  Duration _position = Duration.zero;
  String? _error;

  // Getters
  Podcast? get currentPodcast => _currentPodcast;
  bool get isPlaying => _isPlaying;
  bool get isLoading => _isLoading;
  Duration get duration => _duration;
  Duration get position => _position;
  String? get error => _error;
  double get progress => _duration.inMilliseconds > 0 
      ? _position.inMilliseconds / _duration.inMilliseconds 
      : 0.0;

  PodcastPlayerService() {
    _setupAudioPlayerListeners();
  }

  void _setupAudioPlayerListeners() {
    // Listener para mudanças de estado
    _audioPlayer.playerStateStream.listen((state) {
      _isPlaying = state.playing;
      _isLoading = state.processingState == ProcessingState.loading ||
                   state.processingState == ProcessingState.buffering;
      notifyListeners();
    });

    // Listener para duração
    _audioPlayer.durationStream.listen((duration) {
      _duration = duration ?? Duration.zero;
      notifyListeners();
    });

    // Listener para posição
    _audioPlayer.positionStream.listen((position) {
      _position = position;
      notifyListeners();
    });

    // Listener para erros
    _audioPlayer.playbackEventStream.listen((event) {}, 
      onError: (Object e, StackTrace stackTrace) {
        _error = 'Erro ao reproduzir: $e';
        _isLoading = false;
        notifyListeners();
        debugPrint('Erro no player: $e');
      }
    );
  }

  Future<void> playPodcast(Podcast podcast) async {
    try {
      _error = null;
      _isLoading = true;
      notifyListeners();

      // Se é o mesmo podcast e está pausado, apenas resume
      if (_currentPodcast?.id == podcast.id && !_isPlaying) {
        await _audioPlayer.play();
        return;
      }

      // Para a reprodução atual se houver
      if (_isPlaying) {
        await _audioPlayer.stop();
      }

      // Define o novo podcast
      _currentPodcast = podcast;
      
      debugPrint('🎙️ Reproduzindo podcast: ${podcast.title}');
      debugPrint('🔗 URL: ${podcast.audioUrl}');

      // Carrega e reproduz o áudio
      await _audioPlayer.setUrl(podcast.audioUrl);
      await _audioPlayer.play();

    } catch (e) {
      _error = 'Erro ao carregar podcast: $e';
      _isLoading = false;
      notifyListeners();
      debugPrint('💥 Erro ao reproduzir podcast: $e');
    }
  }

  Future<void> togglePlayPause() async {
    try {
      if (_audioPlayer.playing) {
        await _audioPlayer.pause();
      } else {
        await _audioPlayer.play();
      }
    } catch (e) {
      _error = 'Erro ao pausar/reproduzir: $e';
      notifyListeners();
    }
  }

  Future<void> stop() async {
    try {
      await _audioPlayer.stop();
      _currentPodcast = null;
      _position = Duration.zero;
      notifyListeners();
    } catch (e) {
      debugPrint('Erro ao parar: $e');
    }
  }

  Future<void> seek(Duration position) async {
    try {
      await _audioPlayer.seek(position);
    } catch (e) {
      debugPrint('Erro ao buscar posição: $e');
    }
  }

  void clearError() {
    _error = null;
    notifyListeners();
  }

  String formatDuration(Duration duration) {
    String twoDigits(int n) => n.toString().padLeft(2, '0');
    final minutes = twoDigits(duration.inMinutes.remainder(60));
    final seconds = twoDigits(duration.inSeconds.remainder(60));
    return '$minutes:$seconds';
  }

  @override
  void dispose() {
    _audioPlayer.dispose();
    super.dispose();
  }
}