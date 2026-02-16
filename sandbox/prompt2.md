<?php

namespace App\Http\Controllers;

use App\Jobs\PrepareCompositionPayloadJob;
use App\Models\MasterSlip;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Log;

class CompositionSlipController extends Controller
{
    /**
     * Orchestrate composition slips generation
     *
     * @param Request $request
     * @param int $baseMasterSlipId
     * @param int $optimizedMasterSlipId
     * @return JsonResponse
     */
    public function orchestrateComposition(
        Request $request,
        int $baseMasterSlipId,
        int $optimizedMasterSlipId
    ): JsonResponse {
        try {
            Log::info('Composition orchestration initiated', [
                'base_master_slip_id' => $baseMasterSlipId,
                'optimized_master_slip_id' => $optimizedMasterSlipId,
            ]);

            // Validate master slips exist
            $this->validateMasterSlipsExist($baseMasterSlipId, $optimizedMasterSlipId);

            // Dispatch job to prepare payload
            PrepareCompositionPayloadJob::dispatch(
                $baseMasterSlipId,
                $optimizedMasterSlipId,
                $this->extractConfigOverride($request)
            );

            return response()->json([
                'status' => 'orchestrated',
                'message' => 'Composition payload preparation queued',
                'data' => [
                    'base_master_slip_id' => $baseMasterSlipId,
                    'optimized_master_slip_id' => $optimizedMasterSlipId,
                    'requested_at' => now()->toISOString(),
                ]
            ], 202);

        } catch (\Exception $e) {
            Log::error('Composition orchestration failed', [
                'error' => $e->getMessage(),
                'trace' => $e->getTraceAsString(),
                'base_id' => $baseMasterSlipId,
                'optimized_id' => $optimizedMasterSlipId,
            ]);

            return response()->json([
                'status' => 'error',
                'message' => 'Composition orchestration failed: ' . $e->getMessage(),
                'error_code' => 'COMPOSITION_ORCHESTRATION_FAILED',
            ], 400);
        }
    }

    /**
     * Validate both master slips exist
     *
     * @param int $baseMasterSlipId
     * @param int $optimizedMasterSlipId
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException
     */
    private function validateMasterSlipsExist(int $baseMasterSlipId, int $optimizedMasterSlipId): void
    {
        MasterSlip::findOrFail($baseMasterSlipId);
        MasterSlip::findOrFail($optimizedMasterSlipId);
        
        Log::debug('Master slips validated', [
            'base_master_slip_id' => $baseMasterSlipId,
            'optimized_master_slip_id' => $optimizedMasterSlipId,
        ]);
    }

    /**
     * Extract config override from request if provided
     *
     * @param Request $request
     * @return array
     */
    private function extractConfigOverride(Request $request): array
    {
        return $request->validate([
            'config.targets.count' => 'sometimes|integer|min:1|max:100',
            'config.targets.min_matches' => 'sometimes|integer|min:2|max:8',
            'config.targets.max_matches' => 'sometimes|integer|min:2|max:8',
            'config.time_clustering.window_minutes' => 'sometimes|integer|min:60|max:360',
            'config.time_clustering.min_gap_minutes' => 'sometimes|integer|min:30|max:240',
            'config.odds_bands.low_min' => 'sometimes|numeric|min:1.01|max:3.0',
            'config.odds_bands.low_max' => 'sometimes|numeric|min:1.01|max:3.0',
            'config.odds_bands.mid_min' => 'sometimes|numeric|min:1.01|max:5.0',
            'config.odds_bands.mid_max' => 'sometimes|numeric|min:1.01|max:5.0',
        ]);
    }
}

Service (App/Services/CompositionSlipPayloadService.php)

<?php

namespace App\Services;

use App\Models\GeneratedSlip;
use Illuminate\Support\Collection;
use Illuminate\Support\Facades\Log;
use InvalidArgumentException;

class CompositionSlipPayloadService
{
    private const REQUIRED_SLIP_COUNT = 50;

    /**
     * Build composition payload from two master slips
     *
     * @param int $baseMasterSlipId
     * @param int $optimizedMasterSlipId
     * @param array $configOverride
     * @return array
     * @throws \RuntimeException
     */
    public function buildPayload(
        int $baseMasterSlipId,
        int $optimizedMasterSlipId,
        array $configOverride = []
    ): array {
        Log::info('Building composition payload', [
            'base_master_slip_id' => $baseMasterSlipId,
            'optimized_master_slip_id' => $optimizedMasterSlipId,
        ]);

        $baseSlips = $this->loadSlips($baseMasterSlipId);
        $optimizedSlips = $this->loadSlips($optimizedMasterSlipId);

        $this->validateSlipCounts($baseSlips, $optimizedSlips);

        $normalizedBaseSlips = $this->normalizeSlips($baseSlips);
        $normalizedOptimizedSlips = $this->normalizeSlips($optimizedSlips);

        Log::debug('Slips normalized', [
            'base_slips_count' => count($normalizedBaseSlips),
            'optimized_slips_count' => count($normalizedOptimizedSlips),
            'total_legs_base' => array_sum(array_map(fn($slip) => count($slip['legs']), $normalizedBaseSlips)),
            'total_legs_optimized' => array_sum(array_map(fn($slip) => count($slip['legs']), $normalizedOptimizedSlips)),
        ]);

        return $this->assemblePayload(
            $normalizedBaseSlips,
            $normalizedOptimizedSlips,
            $configOverride
        );
    }

    /**
     * Load slips for a master slip with legs
     *
     * @param int $masterSlipId
     * @return Collection
     */
    private function loadSlips(int $masterSlipId): Collection
    {
        $slips = GeneratedSlip::with('legs')
            ->where('master_slip_id', $masterSlipId)
            ->orderByDesc('confidence_score')
            ->limit(self::REQUIRED_SLIP_COUNT)
            ->get();

        if ($slips->isEmpty()) {
            Log::warning('No slips found for master slip', [
                'master_slip_id' => $masterSlipId,
            ]);
        }

        return $slips;
    }

    /**
     * Validate slip counts meet requirements
     *
     * @param Collection $baseSlips
     * @param Collection $optimizedSlips
     * @throws \RuntimeException
     */
    private function validateSlipCounts(Collection $baseSlips, Collection $optimizedSlips): void
    {
        if ($baseSlips->count() < self::REQUIRED_SLIP_COUNT) {
            throw new \RuntimeException(sprintf(
                'Insufficient base slips. Required: %d, Found: %d',
                self::REQUIRED_SLIP_COUNT,
                $baseSlips->count()
            ));
        }

        if ($optimizedSlips->count() < self::REQUIRED_SLIP_COUNT) {
            throw new \RuntimeException(sprintf(
                'Insufficient optimized slips. Required: %d, Found: %d',
                self::REQUIRED_SLIP_COUNT,
                $optimizedSlips->count()
            ));
        }

        Log::debug('Slip counts validated', [
            'base_slips_count' => $baseSlips->count(),
            'optimized_slips_count' => $optimizedSlips->count(),
        ]);
    }

    /**
     * Normalize slips into required structure
     *
     * @param Collection $slips
     * @return array
     */
    private function normalizeSlips(Collection $slips): array
    {
        return $slips->take(self::REQUIRED_SLIP_COUNT)
            ->map(function (GeneratedSlip $slip) {
                return [
                    'master_slip_id' => $slip->master_slip_id,
                    'slip_id' => $slip->slip_id,
                    'total_odds' => $slip->total_odds,
                    'confidence_score' => $slip->confidence_score,
                    'risk_level' => $slip->risk_level,
                    'legs' => $this->normalizeLegs($slip->legs),
                ];
            })
            ->values()
            ->toArray();
    }

    /**
     * Normalize slip legs into required structure
     *
     * @param Collection $legs
     * @return array
     */
    private function normalizeLegs(Collection $legs): array
    {
        return $legs->map(function ($leg) {
            return [
                'match_id' => $leg->match_id,
                'market' => $leg->market,
                'selection' => $leg->selection,
                'odds' => $leg->odds,
                'kickoff_time' => $leg->kickoff_time,
                'league' => $leg->league,
                'home_team' => $leg->home_team,
                'away_team' => $leg->away_team,
            ];
        })->toArray();
    }

    /**
     * Assemble final payload
     *
     * @param array $baseSlips
     * @param array $optimizedSlips
     * @param array $configOverride
     * @return array
     */
    private function assemblePayload(array $baseSlips, array $optimizedSlips, array $configOverride): array
    {
        $payload = [
            'master_slip' => [
                'id' => $this->generateCompositionMasterId($baseSlips, $optimizedSlips),
                'composition_slips' => array_merge($this->getDefaultConfig(), $configOverride),
            ],
            'base_slips' => $baseSlips,
            'optimized_slips' => $optimizedSlips,
        ];

        Log::debug('Payload assembled', [
            'payload_size_bytes' => strlen(json_encode($payload)),
            'composition_master_id' => $payload['master_slip']['id'],
        ]);

        return $payload;
    }

    /**
     * Generate deterministic composition master ID
     *
     * @param array $baseSlips
     * @param array $optimizedSlips
     * @return int
     */
    private function generateCompositionMasterId(array $baseSlips, array $optimizedSlips): int
    {
        // Create deterministic ID from slip IDs
        $baseIds = array_column($baseSlips, 'slip_id');
        $optimizedIds = array_column($optimizedSlips, 'slip_id');
        
        $combined = array_merge($baseIds, $optimizedIds);
        sort($combined);
        
        return crc32(implode('_', $combined));
    }

    /**
     * Get default configuration for composition
     *
     * @return array
     */
    private function getDefaultConfig(): array
    {
        return [
            'targets' => [
                'count' => 50,
                'min_matches' => 2,
                'max_matches' => 4,
            ],
            'time_clustering' => [
                'window_minutes' => 120,
                'min_gap_minutes' => 90,
            ],
            'odds_bands' => [
                'low_min' => 1.20,
                'low_max' => 1.40,
                'mid_min' => 2.00,
                'mid_max' => 2.60,
            ],
        ];
    }
}

Job: Prepare Composition Payload (App/Jobs/PrepareCompositionPayloadJob.php)

<?php

namespace App\Jobs;

use App\Services\CompositionSlipPayloadService;
use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Bus\Dispatchable;
use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Queue\SerializesModels;
use Illuminate\Support\Facades\Log;

class PrepareCompositionPayloadJob implements ShouldQueue
{
    use Dispatchable, InteractsWithQueue, Queueable, SerializesModels;

    public int $tries = 3;
    public int $timeout = 120;

    public function __construct(
        private int $baseMasterSlipId,
        private int $optimizedMasterSlipId,
        private array $configOverride = []
    ) {
        $this->onQueue('composition');
    }

    public function handle(CompositionSlipPayloadService $service): void
    {
        Log::info('PrepareCompositionPayloadJob started', [
            'base_master_slip_id' => $this->baseMasterSlipId,
            'optimized_master_slip_id' => $this->optimizedMasterSlipId,
            'job_id' => $this->job->getJobId(),
        ]);

        try {
            $payload = $service->buildPayload(
                $this->baseMasterSlipId,
                $this->optimizedMasterSlipId,
                $this->configOverride
            );

            Log::info('Payload prepared successfully', [
                'composition_master_id' => $payload['master_slip']['id'],
                'base_slips_count' => count($payload['base_slips']),
                'optimized_slips_count' => count($payload['optimized_slips']),
            ]);

            // Dispatch to Python engine
            DispatchCompositionToPythonJob::dispatch($payload);

            Log::info('DispatchCompositionToPythonJob queued', [
                'composition_master_id' => $payload['master_slip']['id'],
            ]);

        } catch (\Exception $e) {
            Log::error('PrepareCompositionPayloadJob failed', [
                'error' => $e->getMessage(),
                'trace' => $e->getTraceAsString(),
                'base_master_slip_id' => $this->baseMasterSlipId,
                'optimized_master_slip_id' => $this->optimizedMasterSlipId,
            ]);

            $this->fail($e);
        }
    }

    public function failed(\Throwable $exception): void
    {
        Log::critical('PrepareCompositionPayloadJob failed permanently', [
            'error' => $exception->getMessage(),
            'base_master_slip_id' => $this->baseMasterSlipId,
            'optimized_master_slip_id' => $this->optimizedMasterSlipId,
        ]);
    }
}

Job: Dispatch to Python (App/Jobs/DispatchCompositionToPythonJob.php)

<?php

namespace App\Jobs;

use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Bus\Dispatchable;
use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Queue\SerializesModels;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;

class DispatchCompositionToPythonJob implements ShouldQueue
{
    use Dispatchable, InteractsWithQueue, Queueable, SerializesModels;

    public int $tries = 3;
    public int $timeout = 60;
    public int $backoff = 30;

    public function __construct(
        private array $payload
    ) {
        $this->onQueue('python-dispatch');
    }

    public function handle(): void
    {
        $compositionMasterId = $this->payload['master_slip']['id'] ?? 'unknown';
        
        Log::info('DispatchCompositionToPythonJob started', [
            'composition_master_id' => $compositionMasterId,
            'payload_size_bytes' => strlen(json_encode($this->payload)),
        ]);

        try {
            $response = Http::timeout(45)
                ->retry(2, 1000)
                ->withHeaders([
                    'Content-Type' => 'application/json',
                    'X-Request-ID' => uniqid('comp-', true),
                ])
                ->post(config('services.python_slip_engine.url') . '/composition-slips/build', $this->payload);

            if ($response->successful()) {
                Log::info('Composition request successful', [
                    'composition_master_id' => $compositionMasterId,
                    'response_status' => $response->status(),
                    'response_body' => $response->json(),
                ]);

                // Optionally dispatch job to store results
                // StoreCompositionResultsJob::dispatch($response->json());
            } else {
                Log::error('Composition request failed', [
                    'composition_master_id' => $compositionMasterId,
                    'response_status' => $response->status(),
                    'response_body' => $response->body(),
                ]);

                $this->fail(new \RuntimeException(
                    "Python engine responded with status: {$response->status()}"
                ));
            }

        } catch (\Illuminate\Http\Client\ConnectionException $e) {
            Log::error('Python engine connection failed', [
                'composition_master_id' => $compositionMasterId,
                'error' => $e->getMessage(),
            ]);

            $this->fail($e);
        } catch (\Exception $e) {
            Log::error('DispatchCompositionToPythonJob failed', [
                'composition_master_id' => $compositionMasterId,
                'error' => $e->getMessage(),
                'trace' => $e->getTraceAsString(),
            ]);

            $this->fail($e);
        }
    }

    public function failed(\Throwable $exception): void
    {
        Log::critical('DispatchCompositionToPythonJob failed permanently', [
            'composition_master_id' => $this->payload['master_slip']['id'] ?? 'unknown',
            'error' => $exception->getMessage(),
            'payload_keys' => array_keys($this->payload),
        ]);
    }
}

 Route Definition (routes/api.php)
<?php

use App\Http\Controllers\CompositionSlipController;
use Illuminate\Support\Facades\Route;

// Composition slips orchestration
Route::post('/composition-slips/{baseMasterSlipId}/{optimizedMasterSlipId}', 
    [CompositionSlipController::class, 'orchestrateComposition'])
    ->name('composition-slips.orchestrate')
    ->whereNumber(['baseMasterSlipId', 'optimizedMasterSlipId']);


Migration for Tracking (Optional)

<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('composition_orchestrations', function (Blueprint $table) {
            $table->id();
            $table->unsignedBigInteger('composition_master_id');
            $table->unsignedBigInteger('base_master_slip_id');
            $table->unsignedBigInteger('optimized_master_slip_id');
            $table->json('payload_metadata')->nullable();
            $table->string('status')->default('pending');
            $table->json('error_log')->nullable();
            $table->timestamp('dispatched_at')->nullable();
            $table->timestamp('completed_at')->nullable();
            $table->timestamps();

            $table->index('composition_master_id');
            $table->index('base_master_slip_id');
            $table->index('optimized_master_slip_id');
            $table->index('status');
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('composition_orchestrations');
    }
};


7. Environment Variables (.env.example)

# Python Slip Engine
PYTHON_SLIP_ENGINE_URL=http://localhost:5000
PYTHON_SLIP_ENGINE_TIMEOUT=45
PYTHON_SLIP_ENGINE_RETRY_ATTEMPTS=2


Service Configuration (config/services.php)
'python_slip_engine' => [
    'url' => env('PYTHON_SLIP_ENGINE_URL', 'http://localhost:8000'),
    'timeout' => env('PYTHON_SLIP_ENGINE_TIMEOUT', 45),
    'retry_attempts' => env('PYTHON_SLIP_ENGINE_RETRY_ATTEMPTS', 2),
],





