import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:just_audio/just_audio.dart';
import 'package:share_plus/share_plus.dart';
import 'package:http/http.dart' as http;
import '../models/song_history.dart';

class RadioService extends ChangeNotifier {
  final AudioPlayer _audioPlayer = AudioPlayer();
  static const String streamUrl = 'https://live9.livemus.com.br:27076/stream';
  static const String metadataApiUrl = 'https://www.radioentrerios.com.br/wp-content/noticias/radio_metadata_api.php';
  static const bool debugMode = kDebugMode; // Ativa logs apenas em modo debug

  bool _isPlaying = false;
  bool _isLoading = false;
  String? _currentProgram;
  String? _currentSong;
  String? _currentArtist;
  String? _currentCover;
  String? _error;
  Timer? _metadataTimer;

  bool get isPlaying => _isPlaying;
  bool get isLoading => _isLoading;
  String? get currentProgram => _currentProgram;
  String? get currentSong => _currentSong;
  String? get currentArtist => _currentArtist;
  String? get currentCover => _currentCover;
  String? get error => _error;

  RadioService() {
    _initPlayer();
  }

  Future<void> _initPlayer() async {
    try {
      // Configura o stream de áudio
      await _audioPlayer.setUrl(streamUrl);
      
      // Escuta mudanças no estado do player
      _audioPlayer.playerStateStream.listen((state) {
        _isPlaying = state.playing;
        _isLoading = state.processingState == ProcessingState.loading ||
                    state.processingState == ProcessingState.buffering;
        notifyListeners();
      });

      // Escuta erros
      _audioPlayer.playbackEventStream.listen(
        (event) {},
        onError: (Object e, StackTrace stackTrace) {
          _error = 'Erro ao reproduzir: $e';
          _isPlaying = false;
          notifyListeners();
        },
      );

      // Iniciar consulta periódica à API de metadados
      _startMetadataPolling();
    } catch (e) {
      _error = 'Erro ao inicializar player: $e';
      notifyListeners();
    }
  }

  Future<void> play() async {
    try {
      _error = null;
      await _audioPlayer.play();
      _updateCurrentProgram();
    } catch (e) {
      _error = 'Erro ao tocar: $e';
      notifyListeners();
    }
  }

  Future<void> pause() async {
    try {
      await _audioPlayer.pause();
    } catch (e) {
      _error = 'Erro ao pausar: $e';
      notifyListeners();
    }
  }

  Future<void> togglePlay() async {
    if (_isPlaying) {
      await pause();
    } else {
      await play();
    }
  }

  void _updateCurrentProgram() {
    // Só mostra a frequência da rádio - sem inventar nomes de programas
    _currentProgram = '105.5 FM';
    notifyListeners();
  }

  // Consulta a API de metadados para obter informações da música atual
  Future<void> _fetchCurrentSong() async {
    try {
      final response = await http.get(
        Uri.parse('$metadataApiUrl?action=current&t=${DateTime.now().millisecondsSinceEpoch}'),
      );

      if (response.statusCode == 200) {
        final result = jsonDecode(response.body);

        if (debugMode) {
          debugPrint('=== API Metadata recebido ===');
          debugPrint('Response: ${response.body}');
        }

        if (result['success'] == true && result['data'] != null) {
          final metadata = result['data'];
          final String title = metadata['title'] ?? '';
          final String artist = metadata['artist'] ?? '';
          final String? cover = metadata['cover'];

          // Só atualiza se for uma música válida (não "Rádio Entre Rios")
          if (artist.isNotEmpty && artist != 'Rádio Entre Rios' && title.isNotEmpty) {
            _currentSong = title;
            _currentArtist = artist;
            _currentCover = cover;

            if (debugMode) {
              debugPrint('Música atualizada: $_currentArtist - $_currentSong');
              debugPrint('Capa: $_currentCover');
            }

            notifyListeners();
          }
        }
      }
    } catch (e) {
      if (debugMode) {
        debugPrint('Erro ao buscar metadados da API: $e');
      }
    }
  }

  // Inicia polling periódico de metadados (a cada 10 segundos)
  void _startMetadataPolling() {
    // Busca imediatamente
    _fetchCurrentSong();

    // Atualiza a cada 10 segundos
    _metadataTimer = Timer.periodic(const Duration(seconds: 10), (timer) {
      _fetchCurrentSong();
    });

    if (debugMode) {
      debugPrint('Polling de metadados iniciado');
    }
  }

  // Para o polling de metadados
  void _stopMetadataPolling() {
    _metadataTimer?.cancel();
    _metadataTimer = null;

    if (debugMode) {
      debugPrint('Polling de metadados parado');
    }
  }

  // Busca o histórico de músicas tocadas
  Future<List<SongHistory>> fetchHistory() async {
    try {
      final response = await http.get(
        Uri.parse('$metadataApiUrl?action=history&t=${DateTime.now().millisecondsSinceEpoch}'),
      );

      if (response.statusCode == 200) {
        final result = jsonDecode(response.body);

        if (debugMode) {
          debugPrint('=== Histórico recebido ===');
          debugPrint('Response: ${response.body}');
        }

        if (result['success'] == true && result['data'] != null) {
          final List<dynamic> data = result['data'];
          return data.map((item) => SongHistory.fromJson(item)).toList();
        }
      }
      return [];
    } catch (e) {
      if (debugMode) {
        debugPrint('Erro ao buscar histórico: $e');
      }
      return [];
    }
  }

  // Busca a letra da música atual
  Future<String?> fetchLyrics(String artist, String title) async {
    try {
      final response = await http.get(
        Uri.parse('$metadataApiUrl?action=lyrics&artist=${Uri.encodeComponent(artist)}&title=${Uri.encodeComponent(title)}'),
      );

      if (response.statusCode == 200) {
        final result = jsonDecode(response.body);

        if (debugMode) {
          debugPrint('=== Letra recebida ===');
          debugPrint('Response: ${response.body}');
        }

        if (result['success'] == true && result['data'] != null) {
          final data = result['data'];
          // Retorna o texto completo da letra
          if (data['text'] != null) {
            return data['text'];
          } else if (data['lines'] != null) {
            final List<dynamic> lines = data['lines'];
            return lines.join('\n');
          }
        }
      }
      return null;
    } catch (e) {
      if (debugMode) {
        debugPrint('Erro ao buscar letra: $e');
      }
      return null;
    }
  }

  Future<void> shareRadio() async {
    await Share.share(
      '🎵 Estou ouvindo Rádio Entre Rios 105.5 FM! \n'
      'Baixe o app: https://radioentrerios.com.br/app',
      subject: 'Rádio Entre Rios 105.5 FM',
    );
  }

  @override
  void dispose() {
    _stopMetadataPolling();
    _audioPlayer.dispose();
    super.dispose();
  }
}