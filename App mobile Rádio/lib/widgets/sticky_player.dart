import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cached_network_image/cached_network_image.dart';
import '../services/radio_service.dart';
import '../services/theme_service.dart';
import '../utils/theme.dart';
import '../screens/song_history_screen.dart';
import '../screens/lyrics_screen.dart';

class StickyPlayer extends StatelessWidget {
  const StickyPlayer({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer2<RadioService, ThemeService>(
      builder: (context, radioService, themeService, child) {
        return SafeArea(
          top: false,
          child: Container(
            height: 80,
            decoration: BoxDecoration(
              color: themeService.isDarkMode 
                ? AppTheme.darkSurface 
                : AppTheme.lightSurface,
              border: Border(
                top: BorderSide(
                  color: themeService.isDarkMode 
                    ? AppTheme.darkDivider 
                    : AppTheme.lightDivider,
                  width: 1,
                ),
              ),
            ),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
            child: Row(
              children: [
                // Logo da rádio ou capa do álbum
                Container(
                  width: 50,
                  height: 50,
                  decoration: BoxDecoration(
                    color: AppTheme.primaryOrange,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(8),
                    child: radioService.currentCover != null && radioService.currentCover!.isNotEmpty
                      ? CachedNetworkImage(
                          imageUrl: radioService.currentCover!,
                          fit: BoxFit.cover,
                          placeholder: (context, url) => const Icon(
                            Icons.radio,
                            color: Colors.white,
                            size: 28,
                          ),
                          errorWidget: (context, url, error) => const Icon(
                            Icons.radio,
                            color: Colors.white,
                            size: 28,
                          ),
                        )
                      : const Icon(
                          Icons.radio,
                          color: Colors.white,
                          size: 28,
                        ),
                  ),
                ),
                const SizedBox(width: 12),
                
                // Informações da rádio
                Expanded(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        radioService.currentProgram ?? '105.5 FM',
                        style: Theme.of(context).textTheme.titleMedium,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                      if (radioService.currentSong != null && radioService.currentArtist != null) ...[
                        const SizedBox(height: 2),
                        Text(
                          '${radioService.currentArtist} - ${radioService.currentSong}',
                          style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                            color: AppTheme.primaryOrange.withOpacity(0.9),
                            fontWeight: FontWeight.w500,
                          ),
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                        ),
                      ],
                    ],
                  ),
                ),
                
                // "AO VIVO" centralizado
                Text(
                  'AO VIVO',
                  style: Theme.of(context).textTheme.titleLarge?.copyWith(
                    color: AppTheme.primaryOrange,
                    fontWeight: FontWeight.bold,
                    letterSpacing: 1.5,
                  ),
                ),
                
                const SizedBox(width: 16),
                
                // Controles do player
                Row(
                  children: [
                    // Botão de histórico
                    IconButton(
                      icon: const Icon(Icons.history),
                      iconSize: 20,
                      onPressed: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) => const SongHistoryScreen(),
                          ),
                        );
                      },
                      color: themeService.isDarkMode
                        ? AppTheme.darkTextSecondary
                        : AppTheme.lightTextSecondary,
                    ),

                    // Botão de letra (só aparece se tiver música tocando)
                    if (radioService.currentSong != null && radioService.currentArtist != null)
                      IconButton(
                        icon: const Icon(Icons.lyrics),
                        iconSize: 20,
                        onPressed: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (context) => LyricsScreen(
                                artist: radioService.currentArtist!,
                                title: radioService.currentSong!,
                              ),
                            ),
                          );
                        },
                        color: themeService.isDarkMode
                          ? AppTheme.darkTextSecondary
                          : AppTheme.lightTextSecondary,
                      ),

                    // Botão de compartilhar
                    IconButton(
                      icon: const Icon(Icons.share),
                      iconSize: 20,
                      onPressed: radioService.shareRadio,
                      color: themeService.isDarkMode
                        ? AppTheme.darkTextSecondary
                        : AppTheme.lightTextSecondary,
                    ),

                    // Botão Play/Pause
                    Container(
                      decoration: const BoxDecoration(
                        color: AppTheme.primaryOrange,
                        shape: BoxShape.circle,
                      ),
                      child: IconButton(
                        icon: Icon(
                          radioService.isPlaying
                              ? Icons.pause_rounded
                              : Icons.play_arrow_rounded,
                        ),
                        iconSize: 32,
                        onPressed: radioService.togglePlay,
                        color: Colors.white,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      );
      },
    );
  }
}