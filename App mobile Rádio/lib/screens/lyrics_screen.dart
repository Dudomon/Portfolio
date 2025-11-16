import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/radio_service.dart';
import '../services/theme_service.dart';
import '../utils/theme.dart';

class LyricsScreen extends StatefulWidget {
  final String artist;
  final String title;

  const LyricsScreen({
    super.key,
    required this.artist,
    required this.title,
  });

  @override
  State<LyricsScreen> createState() => _LyricsScreenState();
}

class _LyricsScreenState extends State<LyricsScreen> {
  String? _lyrics;
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadLyrics();
  }

  Future<void> _loadLyrics() async {
    setState(() => _isLoading = true);

    final radioService = context.read<RadioService>();
    final lyrics = await radioService.fetchLyrics(widget.artist, widget.title);

    setState(() {
      _lyrics = lyrics;
      _isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    final themeService = context.watch<ThemeService>();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Letra da Música'),
        backgroundColor: themeService.isDarkMode
          ? AppTheme.darkBackground
          : AppTheme.lightBackground,
      ),
      body: _isLoading
        ? const Center(child: CircularProgressIndicator())
        : _lyrics == null
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(
                    Icons.music_off,
                    size: 64,
                    color: AppTheme.primaryOrange.withOpacity(0.5),
                  ),
                  const SizedBox(height: 16),
                  Text(
                    'Letra não encontrada',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      color: themeService.isDarkMode
                        ? Colors.white54
                        : Colors.black54,
                    ),
                  ),
                ],
              ),
            )
          : SingleChildScrollView(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Header com informações da música
                  Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: AppTheme.primaryOrange.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                        color: AppTheme.primaryOrange.withOpacity(0.3),
                      ),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          widget.title,
                          style: Theme.of(context).textTheme.titleLarge?.copyWith(
                            fontWeight: FontWeight.bold,
                            color: AppTheme.primaryOrange,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          widget.artist,
                          style: Theme.of(context).textTheme.titleMedium?.copyWith(
                            color: themeService.isDarkMode
                              ? Colors.white70
                              : Colors.black87,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 24),
                  // Letra da música
                  Text(
                    _lyrics!,
                    style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                      height: 1.8,
                      fontSize: 16,
                    ),
                  ),
                  const SizedBox(height: 40),
                ],
              ),
            ),
    );
  }
}
