import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/podcast_player_service.dart';
import '../services/theme_service.dart';
import '../utils/theme.dart';

class PodcastMiniPlayer extends StatelessWidget {
  const PodcastMiniPlayer({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer2<PodcastPlayerService, ThemeService>(
      builder: (context, playerService, themeService, child) {
        if (playerService.currentPodcast == null) {
          return const SizedBox.shrink();
        }

        return Container(
          height: 80,
          margin: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: themeService.isDarkMode 
              ? AppTheme.darkCard 
              : AppTheme.lightCard,
            borderRadius: BorderRadius.circular(16),
            boxShadow: [
              BoxShadow(
                color: themeService.isDarkMode 
                  ? AppTheme.primaryOrange.withOpacity(0.1)
                  : Colors.black.withOpacity(0.1),
                blurRadius: 8,
                offset: const Offset(0, 4),
              ),
            ],
          ),
          child: Column(
            children: [
              // Progress bar
              if (playerService.duration.inMilliseconds > 0)
                LinearProgressIndicator(
                  value: playerService.progress,
                  backgroundColor: themeService.isDarkMode 
                    ? AppTheme.darkDivider 
                    : AppTheme.lightDivider,
                  valueColor: const AlwaysStoppedAnimation<Color>(AppTheme.primaryOrange),
                  minHeight: 2,
                ),
              
              // Player content
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  child: Row(
                    children: [
                      // Podcast icon
                      Container(
                        width: 48,
                        height: 48,
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            colors: [
                              AppTheme.primaryOrange,
                              AppTheme.primaryOrange.withOpacity(0.8),
                            ],
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                          ),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: const Icon(
                          Icons.podcasts_rounded,
                          color: Colors.white,
                          size: 24,
                        ),
                      ),
                      const SizedBox(width: 12),
                      
                      // Podcast info
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Text(
                              playerService.currentPodcast!.title,
                              style: Theme.of(context).textTheme.titleSmall?.copyWith(
                                fontWeight: FontWeight.w600,
                              ),
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                            ),
                            const SizedBox(height: 2),
                            Row(
                              children: [
                                Text(
                                  playerService.formatDuration(playerService.position),
                                  style: Theme.of(context).textTheme.labelSmall?.copyWith(
                                    color: AppTheme.primaryOrange,
                                    fontWeight: FontWeight.w500,
                                  ),
                                ),
                                Text(
                                  ' / ${playerService.formatDuration(playerService.duration)}',
                                  style: Theme.of(context).textTheme.labelSmall?.copyWith(
                                    color: themeService.isDarkMode 
                                      ? AppTheme.darkTextSecondary 
                                      : AppTheme.lightTextSecondary,
                                  ),
                                ),
                              ],
                            ),
                          ],
                        ),
                      ),
                      
                      // Controls
                      Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          // Stop button
                          IconButton(
                            icon: const Icon(Icons.stop_rounded),
                            iconSize: 20,
                            onPressed: playerService.stop,
                            color: themeService.isDarkMode 
                              ? AppTheme.darkTextSecondary 
                              : AppTheme.lightTextSecondary,
                          ),
                          
                          // Play/Pause button
                          Container(
                            decoration: const BoxDecoration(
                              color: AppTheme.primaryOrange,
                              shape: BoxShape.circle,
                            ),
                            child: IconButton(
                              icon: Icon(
                                playerService.isPlaying 
                                  ? Icons.pause_rounded 
                                  : Icons.play_arrow_rounded,
                              ),
                              iconSize: 24,
                              onPressed: playerService.togglePlayPause,
                              color: Colors.white,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}