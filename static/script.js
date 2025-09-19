/**
 * LLM Firewall API Configuration UI
 * Modern, accessible configuration interface
 */

class ConfigUI {
    constructor() {
        this.currentConfig = {};
        this.isDirty = false;
        this.init();
    }

    /**
     * Initialize the configuration UI
     */
    init() {
        console.log('ConfigUI initializing...'); // Debug log
        this.setupTabNavigation();
        this.setupFormHandlers();
        this.setupRangeInputs();
        this.setupCheckboxGroups();
        this.loadConfiguration();
        
        // Set up keyboard shortcuts
        document.addEventListener('keydown', this.handleKeyboardShortcuts.bind(this));
        
        // Warn user about unsaved changes
        window.addEventListener('beforeunload', this.handleBeforeUnload.bind(this));
        
        // Set up configuration management for the Forwarding tab
        this.setupConfigurationManagement();
        
        console.log('ConfigUI initialization complete'); // Debug log
    }

    /**
     * Set up accessible tab navigation
     */
    setupTabNavigation() {
        const tabButtons = document.querySelectorAll('.tab-button');
        const tabPanels = document.querySelectorAll('.tab-panel');

        // Find the initially active tab
        let activeIndex = 0;
        tabButtons.forEach((button, index) => {
            if (button.classList.contains('active') || button.getAttribute('aria-selected') === 'true') {
                activeIndex = index;
            }
        });

        // Set up the initial state
        this.switchTab(activeIndex);

        tabButtons.forEach((button, index) => {
            button.addEventListener('click', () => {
                this.switchTab(index);
            });

            button.addEventListener('keydown', (e) => {
                let targetIndex = index;
                
                switch (e.key) {
                    case 'ArrowRight':
                    case 'ArrowDown':
                        targetIndex = (index + 1) % tabButtons.length;
                        break;
                    case 'ArrowLeft':
                    case 'ArrowUp':
                        targetIndex = (index - 1 + tabButtons.length) % tabButtons.length;
                        break;
                    case 'Home':
                        targetIndex = 0;
                        break;
                    case 'End':
                        targetIndex = tabButtons.length - 1;
                        break;
                    default:
                        return; // Don't handle other keys
                }
                
                e.preventDefault();
                this.switchTab(targetIndex);
                tabButtons[targetIndex].focus();
            });
        });
    }

    /**
     * Switch to a specific tab
     */
    switchTab(index) {
        const tabButtons = document.querySelectorAll('.tab-button');
        const tabPanels = document.querySelectorAll('.tab-panel');

        // Update buttons
        tabButtons.forEach((button, i) => {
            const isSelected = i === index;
            button.setAttribute('aria-selected', isSelected);
            button.setAttribute('tabindex', isSelected ? '0' : '-1');
            
            // Add/remove active class for styling
            if (isSelected) {
                button.classList.add('active');
            } else {
                button.classList.remove('active');
            }
        });

        // Update panels - use CSS class instead of hidden attribute
        tabPanels.forEach((panel, i) => {
            if (i === index) {
                panel.classList.add('active');
            } else {
                panel.classList.remove('active');
            }
        });

        // Announce tab change to screen readers
        const tabName = tabButtons[index].textContent;
        announceToScreenReader(`${tabName} tab selected`);
    }

    /**
     * Set up form handlers
     */
    setupFormHandlers() {
        const form = document.getElementById('configForm');
        const loadBtn = document.getElementById('loadConfigBtn');

        if (form) {
            console.log('Setting up form submit handler');
            form.addEventListener('submit', this.handleFormSubmit.bind(this));
            form.addEventListener('input', this.handleFormInput.bind(this));
        } else {
            console.log('ERROR: Global config form not found!');
        }
        
        if (loadBtn) {
            loadBtn.addEventListener('click', this.loadConfiguration.bind(this));
        }
        
        // Add reload configuration button handler
        const reloadBtn = document.getElementById('reloadConfigBtn');
        if (reloadBtn) {
            reloadBtn.addEventListener('click', this.reloadApplicationConfiguration.bind(this));
        }
        
        // Also add a direct click handler to the save button to trigger form submission
        const saveBtn = document.getElementById('saveGlobalConfigBtn');
        if (saveBtn) {
            console.log('Setting up save button click handler');
            saveBtn.addEventListener('click', (e) => {
                console.log('Save button clicked directly');
                this.updateStatus('Save button clicked, processing configuration...', 'info');
                
                // Prevent any default behavior
                e.preventDefault();
                
                // Find the form and call our handler directly
                const form = document.getElementById('configForm');
                if (form) {
                    console.log('Calling form submit handler directly');
                    // Create a fake event object that our handler expects
                    const fakeEvent = {
                        preventDefault: () => {},
                        target: form
                    };
                    this.handleFormSubmit(fakeEvent);
                } else {
                    console.log('ERROR: Could not find form to submit');
                    this.updateStatus('Error: Configuration form not found', 'error');
                }
            });
        } else {
            console.log('ERROR: Save button not found!');
        }

        // Set up endpoint testing
        // NOTE: This is now handled in the init method to ensure method is available
        // Leaving this comment for reference

        // Set up API key field handlers to detect changes from masked values
        const apiKeyElements = ['hfToken', 'togetherApiKey', 'openaiApiKey'];
        apiKeyElements.forEach(elementId => {
            const element = document.getElementById(elementId);
            if (element) {
                let previousValue = element.value;
                
                // Handle input - only clear masked value when user actually types new content
                element.addEventListener('input', (e) => {
                    const currentValue = e.target.value;
                    const isMasked = e.target.dataset.hasRealValue === "false";
                    const wasMasked = previousValue === '••••••••••••••••••••';
                    
                    // Only clear if user actually added new characters to a masked field
                    if (isMasked && wasMasked && currentValue !== previousValue && !currentValue.includes('•')) {
                        // User typed real content, mark as real value
                        e.target.style.color = '';
                        e.target.placeholder = `Enter new ${elementId === 'hfToken' ? 'Hugging Face token' : 
                                                            elementId === 'togetherApiKey' ? 'Together AI API key' : 
                                                            'OpenAI API key'}`;
                        e.target.dataset.hasRealValue = "true";
                    } else if (isMasked && currentValue.includes('•') && currentValue !== '••••••••••••••••••••') {
                        // User is typing in masked field, clear the bullets and keep their input
                        const cleanValue = currentValue.replace(/•/g, '');
                        e.target.value = cleanValue;
                        e.target.style.color = '';
                        e.target.dataset.hasRealValue = "true";
                    }
                    
                    previousValue = e.target.value;
                });
                
                // Handle blur - if empty and was masked, restore masked value
                element.addEventListener('blur', (e) => {
                    if (e.target.dataset.originalValue === '[REDACTED]' && 
                        e.target.value.trim() === '' && 
                        e.target.dataset.hasRealValue === "true") {
                        // User cleared the field but it was originally masked, restore mask
                        e.target.value = '••••••••••••••••••••';
                        e.target.style.color = '#6b7280';
                        e.target.dataset.hasRealValue = "false";
                        e.target.placeholder = '';
                        previousValue = e.target.value;
                    }
                });
                
                // Initialize previousValue
                previousValue = element.value;
            }
        });

        // LLM Guard enable/disable toggle
        const llmguardEnabled = document.getElementById('llmguardEnabled');
        if (llmguardEnabled) {
            llmguardEnabled.addEventListener('change', this.updateLLMGuardSettingsVisibility.bind(this));
        }
    }

    /**
     * Set up range input value displays
     */
    setupRangeInputs() {
        const rangeInputs = document.querySelectorAll('input[type="range"]');
        
        rangeInputs.forEach(input => {
            const valueDisplay = document.getElementById(input.id + '-value');
            if (valueDisplay) {
                input.addEventListener('input', () => {
                    valueDisplay.textContent = input.value;
                });
            }
        });
    }

    /**
     * Set up checkbox group handlers
     */
    setupCheckboxGroups() {
        // LlamaFirewall scanners
        const llamafirewallScanners = document.querySelectorAll('#panel-llamafirewall input[type="checkbox"]');
        llamafirewallScanners.forEach(checkbox => {
            checkbox.addEventListener('change', this.updateLlamaFirewallConfig.bind(this));
        });

        // LLM Guard input scanners
        const llmguardScanners = document.querySelectorAll('#panel-llmguard .scanner-grid input[type="checkbox"]');
        llmguardScanners.forEach(checkbox => {
            checkbox.addEventListener('change', this.updateLLMGuardScanners.bind(this));
        });

        // Azure AI Content Safety enable/disable toggle
        const azureEnabled = document.getElementById('azureEnabled');
        if (azureEnabled) {
            azureEnabled.addEventListener('change', this.updateAzureSettingsVisibility.bind(this));
        }
    }

    /**
     * Update LlamaFirewall scanner configuration
     */
    updateLlamaFirewallConfig() {
        const checkboxes = document.querySelectorAll('#panel-llamafirewall input[type="checkbox"]:checked');
        const selectedScanners = Array.from(checkboxes).map(cb => cb.value);
        
        const config = {
            "USER": selectedScanners
        };
        
        const hiddenField = document.getElementById('llamafirewallScanners');
        if (hiddenField) {
            hiddenField.value = JSON.stringify(config);
            this.isDirty = true;
        }
    }

    /**
     * Update LLM Guard scanners configuration
     */
    updateLLMGuardScanners() {
        const checkboxes = document.querySelectorAll('#panel-llmguard .scanner-grid input[type="checkbox"]:checked');
        const selectedScanners = Array.from(checkboxes).map(cb => cb.value);
        
        const hiddenField = document.getElementById('llmguardInputScanners');
        if (hiddenField) {
            hiddenField.value = JSON.stringify(selectedScanners);
            this.isDirty = true;
        }
    }

    /**
     * Update LLM Guard settings visibility based on enabled state
     */
    updateLLMGuardSettingsVisibility() {
        const enabled = document.getElementById('llmguardEnabled')?.checked || false;
        const settings = document.getElementById('llmguard-settings');
        
        if (settings) {
            settings.style.opacity = enabled ? '1' : '0.5';
            settings.style.pointerEvents = enabled ? 'auto' : 'none';
            
            // Update ARIA attributes
            const inputs = settings.querySelectorAll('input, select');
            inputs.forEach(input => {
                input.setAttribute('aria-disabled', !enabled);
            });
        }
    }

    /**
     * Update Azure AI Content Safety settings visibility based on enabled state
     */
    updateAzureSettingsVisibility() {
        const enabled = document.getElementById('azureEnabled')?.checked || false;
        const settings = document.getElementById('azure-settings');
        
        if (settings) {
            settings.style.opacity = enabled ? '1' : '0.5';
            settings.style.pointerEvents = enabled ? 'auto' : 'none';
            
            // Update ARIA attributes
            const inputs = settings.querySelectorAll('input, select');
            inputs.forEach(input => {
                input.setAttribute('aria-disabled', !enabled);
            });
        }
    }

    /**
     * Handle form input changes
     */
    handleFormInput() {
        this.isDirty = true;
        this.updateStatus('Configuration modified. Remember to save your changes.', 'warning');
    }

    /**
     * Handle form submission
     */
    async handleFormSubmit(e) {
        e.preventDefault();
        console.log('Form submit handler called');
        this.updateStatus('Form submitted!', 'info');
        
        const form = e.target;
        const formData = new FormData(form);
        const config = Object.fromEntries(formData.entries());
        
        // Parse JSON fields that come as strings for manipulation
        try {
            if (config.LLAMAFIREWALL_SCANNERS) {
                const scannersObj = JSON.parse(config.LLAMAFIREWALL_SCANNERS);
                // Re-stringify to ensure it's a string
                config.LLAMAFIREWALL_SCANNERS = JSON.stringify(scannersObj);
            }
            if (config.LLMGUARD_INPUT_SCANNERS) {
                const scannersObj = JSON.parse(config.LLMGUARD_INPUT_SCANNERS);
                // Re-stringify to ensure it's a string
                config.LLMGUARD_INPUT_SCANNERS = JSON.stringify(scannersObj);
            }
        } catch (error) {
            this.updateStatus('Invalid scanner configuration format.', 'error');
            return;
        }
        
        // Convert checkbox values to proper boolean strings
        // Checkboxes that are checked will have value 'on', unchecked won't be in FormData
        config.LLMGUARD_ENABLED = config.LLMGUARD_ENABLED === 'on' ? 'true' : 'false';
        config.LLMGUARD_FAIL_FAST = config.LLMGUARD_FAIL_FAST === 'on' ? 'true' : 'false';
        config.TOKENIZERS_PARALLELISM = config.TOKENIZERS_PARALLELISM === 'on' ? 'true' : 'false';
        config.AZURE_CONTENT_SAFETY_ENABLED = config.AZURE_CONTENT_SAFETY_ENABLED === 'on' ? 'true' : 'false';
        config.AZURE_CONTENT_SAFETY_TEXT_ENABLED = config.AZURE_CONTENT_SAFETY_TEXT_ENABLED === 'on' ? 'true' : 'false';
        config.AZURE_CONTENT_SAFETY_JAILBREAK_ENABLED = config.AZURE_CONTENT_SAFETY_JAILBREAK_ENABLED === 'on' ? 'true' : 'false';
        
        // Handle API keys - don't send masked values, exclude them if unchanged
        const apiKeyFields = ['HF_TOKEN', 'TOGETHER_API_KEY', 'OPENAI_API_KEY', 'AZURE_CONTENT_SAFETY_KEY'];
        const apiKeyElementIds = ['hfToken', 'togetherApiKey', 'openaiApiKey', 'azureKey'];
        
        // Handle Azure endpoint separately (treated like an API key for masking)
        const sensitiveFields = ['HF_TOKEN', 'TOGETHER_API_KEY', 'OPENAI_API_KEY', 'AZURE_CONTENT_SAFETY_KEY', 'AZURE_CONTENT_SAFETY_ENDPOINT'];
        const sensitiveElementIds = ['hfToken', 'togetherApiKey', 'openaiApiKey', 'azureKey', 'azureEndpoint'];
        
        sensitiveFields.forEach((fieldName, index) => {
            const element = document.getElementById(sensitiveElementIds[index]);
            if (element) {
                const isMasked = element.dataset.hasRealValue === "false";
                const currentValue = config[fieldName];
                
                if (isMasked && currentValue === '••••••••••••••••••••') {
                    // This is a masked value that hasn't been changed, exclude it from config
                    delete config[fieldName];
                    console.log(`DEBUG: Excluded ${fieldName} from save (unchanged masked value)`);
                } else if (currentValue && !isMasked) {
                    // This is a real value (either new or updated)
                    console.log(`DEBUG: Including ${fieldName} in save (real value)`);
                } else if (!currentValue || currentValue.trim() === '') {
                    // Empty value - explicitly set to empty string to clear the key
                    config[fieldName] = '';
                    console.log(`DEBUG: Including ${fieldName} in save (clearing value)`);
                }
            }
        });
        
        // Convert numeric fields to strings (they come as strings from FormData anyway)
        // No need to parseFloat here since the backend expects strings
        
        try {
            this.updateStatus('Saving configuration...', 'info');
            
            const response = await fetch('/api/env-config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ config_data: config })
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP ${response.status}: ${errorText}`);
            }

            this.currentConfig = config;
            this.isDirty = false;
            this.updateStatus('Configuration saved successfully!', 'success');
            announceToScreenReader('Configuration saved successfully');

            // Trigger configuration reload in the backend
            setTimeout(() => this.reloadApplicationConfiguration(), 500);

        } catch (error) {
            console.error('Error saving configuration:', error);
            this.updateStatus(`Failed to save configuration: ${error.message}`, 'error');
        }
    }

    /**
     * Load configuration from server
     */
    async loadConfiguration() {
        try {
            this.updateStatus('Loading configuration...', 'info');
            
            const response = await fetch('/api/env-config');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const responseData = await response.json();
            const config = responseData.config || responseData; // Handle nested config structure
            this.populateForm(config);
            this.currentConfig = config;
            this.isDirty = false;
            this.updateStatus('Configuration loaded successfully!', 'success');
            
        } catch (error) {
            console.error('Error loading configuration:', error);
            this.updateStatus(`Failed to load configuration: ${error.message}`, 'error');
        }
    }

    /**
     * Populate form with configuration data
     */
    populateForm(config) {
        console.log('Populating form with config:', config);
        
        // API Keys (show placeholder for redacted values)
        const hfToken = document.getElementById('hfToken');
        const togetherApiKey = document.getElementById('togetherApiKey');
        const openaiApiKey = document.getElementById('openaiApiKey');
        
        console.log('DEBUG: API Key elements found:', {
            hfToken: !!hfToken,
            togetherApiKey: !!togetherApiKey,
            openaiApiKey: !!openaiApiKey
        });
        
        if (hfToken) {
            if (config.HF_TOKEN && config.HF_TOKEN !== '[REDACTED]') {
                hfToken.value = config.HF_TOKEN;
                hfToken.placeholder = '';
                hfToken.style.fontStyle = "";
                hfToken.style.color = "";
                hfToken.dataset.hasRealValue = "true";
                console.log('DEBUG: Set HF_TOKEN value');
            } else if (config.HF_TOKEN === '[REDACTED]') {
                hfToken.value = '••••••••••••••••••••'; // Masked value
                hfToken.placeholder = '';
                hfToken.style.fontStyle = "normal";
                hfToken.style.color = "#6b7280";
                hfToken.dataset.hasRealValue = "false"; // Track that this is masked
                hfToken.dataset.originalValue = '[REDACTED]'; // Store original state
                console.log('DEBUG: Set HF_TOKEN masked value');
            } else {
                // No API key configured
                hfToken.value = '';
                hfToken.placeholder = "Enter your Hugging Face token";
                hfToken.style.fontStyle = "";
                hfToken.style.color = "";
                hfToken.dataset.hasRealValue = "true";
                console.log('DEBUG: Set HF_TOKEN default placeholder');
            }
        }
        if (togetherApiKey) {
            if (config.TOGETHER_API_KEY && config.TOGETHER_API_KEY !== '[REDACTED]') {
                togetherApiKey.value = config.TOGETHER_API_KEY;
                togetherApiKey.placeholder = '';
                togetherApiKey.style.fontStyle = "";
                togetherApiKey.style.color = "";
                togetherApiKey.dataset.hasRealValue = "true";
                console.log('DEBUG: Set TOGETHER_API_KEY value');
            } else if (config.TOGETHER_API_KEY === '[REDACTED]') {
                togetherApiKey.value = '••••••••••••••••••••'; // Masked value
                togetherApiKey.placeholder = '';
                togetherApiKey.style.fontStyle = "normal";
                togetherApiKey.style.color = "#6b7280";
                togetherApiKey.dataset.hasRealValue = "false"; // Track that this is masked
                togetherApiKey.dataset.originalValue = '[REDACTED]'; // Store original state
                console.log('DEBUG: Set TOGETHER_API_KEY masked value');
            } else {
                // No API key configured
                togetherApiKey.value = '';
                togetherApiKey.placeholder = "Enter your Together AI API key";
                togetherApiKey.style.fontStyle = "";
                togetherApiKey.style.color = "";
                togetherApiKey.dataset.hasRealValue = "true";
                console.log('DEBUG: Set TOGETHER_API_KEY default placeholder');
            }
        }
        if (openaiApiKey) {
            if (config.OPENAI_API_KEY && config.OPENAI_API_KEY !== '[REDACTED]') {
                openaiApiKey.value = config.OPENAI_API_KEY;
                openaiApiKey.placeholder = '';
                openaiApiKey.style.fontStyle = "";
                openaiApiKey.style.color = "";
                openaiApiKey.dataset.hasRealValue = "true";
                console.log('DEBUG: Set OPENAI_API_KEY value');
            } else if (config.OPENAI_API_KEY === '[REDACTED]') {
                openaiApiKey.value = '••••••••••••••••••••'; // Masked value
                openaiApiKey.placeholder = '';
                openaiApiKey.style.fontStyle = "normal";
                openaiApiKey.style.color = "#6b7280";
                openaiApiKey.dataset.hasRealValue = "false"; // Track that this is masked
                openaiApiKey.dataset.originalValue = '[REDACTED]'; // Store original state
                console.log('DEBUG: Set OPENAI_API_KEY masked value');
            } else {
                // No API key configured
                openaiApiKey.value = '';
                openaiApiKey.placeholder = "Enter your OpenAI API key";
                openaiApiKey.style.fontStyle = "";
                openaiApiKey.style.color = "";
                openaiApiKey.dataset.hasRealValue = "true";
                console.log('DEBUG: Set OPENAI_API_KEY default placeholder');
            }
        }
        
        // Performance settings
        const threadPoolWorkers = document.getElementById('threadPoolWorkers');
        if (threadPoolWorkers && config.THREAD_POOL_WORKERS) {
            threadPoolWorkers.value = config.THREAD_POOL_WORKERS;
        }
        
        const logLevel = document.getElementById('logLevel');
        if (logLevel && config.LOG_LEVEL) {
            logLevel.value = config.LOG_LEVEL;
        }
        
        // LlamaFirewall scanners
        if (config.LLAMAFIREWALL_SCANNERS) {
            const hiddenField = document.getElementById('llamafirewallScanners');
            if (hiddenField) {
                hiddenField.value = typeof config.LLAMAFIREWALL_SCANNERS === 'string' 
                    ? config.LLAMAFIREWALL_SCANNERS 
                    : JSON.stringify(config.LLAMAFIREWALL_SCANNERS);
            }
            
            // Update checkboxes
            const scannerConfig = typeof config.LLAMAFIREWALL_SCANNERS === 'string' 
                ? JSON.parse(config.LLAMAFIREWALL_SCANNERS) 
                : config.LLAMAFIREWALL_SCANNERS;
            const userScanners = scannerConfig?.USER || [];
            document.querySelectorAll('#panel-llamafirewall input[type="checkbox"]').forEach(checkbox => {
                checkbox.checked = userScanners.includes(checkbox.value);
            });
        }
        
        // LLM Guard settings
        const llmguardEnabled = document.getElementById('llmguardEnabled');
        if (llmguardEnabled) {
            llmguardEnabled.checked = config.LLMGUARD_ENABLED === 'true' || config.LLMGUARD_ENABLED === true;
        }
        
        // LLM Guard scanners
        if (config.LLMGUARD_INPUT_SCANNERS) {
            const hiddenField = document.getElementById('llmguardInputScanners');
            if (hiddenField) {
                hiddenField.value = typeof config.LLMGUARD_INPUT_SCANNERS === 'string' 
                    ? config.LLMGUARD_INPUT_SCANNERS 
                    : JSON.stringify(config.LLMGUARD_INPUT_SCANNERS);
            }
            
            // Update checkboxes
            const inputScanners = typeof config.LLMGUARD_INPUT_SCANNERS === 'string' 
                ? JSON.parse(config.LLMGUARD_INPUT_SCANNERS) 
                : config.LLMGUARD_INPUT_SCANNERS;
            document.querySelectorAll('#panel-llmguard .scanner-grid input[type="checkbox"]').forEach(checkbox => {
                checkbox.checked = inputScanners.includes(checkbox.value);
            });
        }
        
        // LLM Guard thresholds and settings
        const toxicityThreshold = document.getElementById('toxicityThreshold');
        const promptInjectionThreshold = document.getElementById('promptInjectionThreshold');
        const sentimentThreshold = document.getElementById('sentimentThreshold');
        const biasThreshold = document.getElementById('biasThreshold');
        const tokenLimit = document.getElementById('tokenLimit');
        const failFast = document.getElementById('failFast');
        
        if (toxicityThreshold) {
            toxicityThreshold.value = config.LLMGUARD_TOXICITY_THRESHOLD || '0.7';
            const display = document.getElementById('toxicityThreshold-value');
            if (display) display.textContent = toxicityThreshold.value;
        }
        
        if (promptInjectionThreshold) {
            promptInjectionThreshold.value = config.LLMGUARD_PROMPT_INJECTION_THRESHOLD || '0.8';
            const display = document.getElementById('promptInjectionThreshold-value');
            if (display) display.textContent = promptInjectionThreshold.value;
        }

        if (sentimentThreshold) {
            sentimentThreshold.value = config.LLMGUARD_SENTIMENT_THRESHOLD || '0.5';
            const display = document.getElementById('sentimentThreshold-value');
            if (display) display.textContent = sentimentThreshold.value;
        }

        if (biasThreshold) {
            biasThreshold.value = config.LLMGUARD_BIAS_THRESHOLD || '0.7';
            const display = document.getElementById('biasThreshold-value');
            if (display) display.textContent = biasThreshold.value;
        }

        if (tokenLimit) {
            tokenLimit.value = config.LLMGUARD_TOKEN_LIMIT || '200';
        }

        if (failFast) {
            failFast.checked = config.LLMGUARD_FAIL_FAST === 'true' || config.LLMGUARD_FAIL_FAST === true;
        }

        // Advanced settings
        const tokenizersParallelism = document.getElementById('tokenizersParallelism');
        if (tokenizersParallelism) {
            tokenizersParallelism.checked = config.TOKENIZERS_PARALLELISM === 'true' || config.TOKENIZERS_PARALLELISM === true;
        }

        // Azure AI Content Safety settings
        const azureEnabled = document.getElementById('azureEnabled');
        if (azureEnabled) {
            azureEnabled.checked = config.AZURE_CONTENT_SAFETY_ENABLED === 'true' || config.AZURE_CONTENT_SAFETY_ENABLED === true;
        }

        const azureTextEnabled = document.getElementById('azureTextEnabled');
        if (azureTextEnabled) {
            azureTextEnabled.checked = config.AZURE_CONTENT_SAFETY_TEXT_ENABLED === 'true' || config.AZURE_CONTENT_SAFETY_TEXT_ENABLED === true;
        }

        const azureJailbreakEnabled = document.getElementById('azureJailbreakEnabled');
        if (azureJailbreakEnabled) {
            azureJailbreakEnabled.checked = config.AZURE_CONTENT_SAFETY_JAILBREAK_ENABLED === 'true' || config.AZURE_CONTENT_SAFETY_JAILBREAK_ENABLED === true;
        }

        const azureEndpoint = document.getElementById('azureEndpoint');
        if (azureEndpoint) {
            if (config.AZURE_CONTENT_SAFETY_ENDPOINT && config.AZURE_CONTENT_SAFETY_ENDPOINT !== '[REDACTED]') {
                azureEndpoint.value = config.AZURE_CONTENT_SAFETY_ENDPOINT;
                azureEndpoint.placeholder = '';
                azureEndpoint.style.fontStyle = "";
                azureEndpoint.style.color = "";
                azureEndpoint.dataset.hasRealValue = "true";
            } else if (config.AZURE_CONTENT_SAFETY_ENDPOINT === '[REDACTED]') {
                azureEndpoint.value = '••••••••••••••••••••'; // Masked value
                azureEndpoint.placeholder = '';
                azureEndpoint.style.fontStyle = "normal";
                azureEndpoint.style.color = "#6b7280";
                azureEndpoint.dataset.hasRealValue = "false";
                azureEndpoint.dataset.originalValue = '[REDACTED]';
            } else {
                azureEndpoint.value = '';
                azureEndpoint.placeholder = "https://your-resource.cognitiveservices.azure.com/";
                azureEndpoint.style.fontStyle = "";
                azureEndpoint.style.color = "";
                azureEndpoint.dataset.hasRealValue = "true";
            }
        }
        
        const azureKey = document.getElementById('azureKey');
        if (azureKey) {
            if (config.AZURE_CONTENT_SAFETY_KEY && config.AZURE_CONTENT_SAFETY_KEY !== '[REDACTED]') {
                azureKey.value = config.AZURE_CONTENT_SAFETY_KEY;
                azureKey.placeholder = '';
                azureKey.style.fontStyle = "";
                azureKey.style.color = "";
                azureKey.dataset.hasRealValue = "true";
            } else if (config.AZURE_CONTENT_SAFETY_KEY === '[REDACTED]') {
                azureKey.value = '••••••••••••••••••••'; // Masked value
                azureKey.placeholder = '';
                azureKey.style.fontStyle = "normal";
                azureKey.style.color = "#6b7280";
                azureKey.dataset.hasRealValue = "false";
                azureKey.dataset.originalValue = '[REDACTED]';
            } else {
                azureKey.value = '';
                azureKey.placeholder = "Enter your Azure Content Safety key";
                azureKey.style.fontStyle = "";
                azureKey.style.color = "";
                azureKey.dataset.hasRealValue = "true";
            }
        }

        // Azure threshold settings
        const azureHateThreshold = document.getElementById('azureHateThreshold');
        if (azureHateThreshold) {
            azureHateThreshold.value = config.AZURE_CONTENT_SAFETY_HATE_THRESHOLD || '4';
            const display = document.getElementById('azureHateThreshold-value');
            if (display) display.textContent = azureHateThreshold.value;
        }

        const azureSelfHarmThreshold = document.getElementById('azureSelfHarmThreshold');
        if (azureSelfHarmThreshold) {
            azureSelfHarmThreshold.value = config.AZURE_CONTENT_SAFETY_SELFHARM_THRESHOLD || '4';
            const display = document.getElementById('azureSelfHarmThreshold-value');
            if (display) display.textContent = azureSelfHarmThreshold.value;
        }

        const azureSexualThreshold = document.getElementById('azureSexualThreshold');
        if (azureSexualThreshold) {
            azureSexualThreshold.value = config.AZURE_CONTENT_SAFETY_SEXUAL_THRESHOLD || '4';
            const display = document.getElementById('azureSexualThreshold-value');
            if (display) display.textContent = azureSexualThreshold.value;
        }

        const azureViolenceThreshold = document.getElementById('azureViolenceThreshold');
        if (azureViolenceThreshold) {
            azureViolenceThreshold.value = config.AZURE_CONTENT_SAFETY_VIOLENCE_THRESHOLD || '4';
            const display = document.getElementById('azureViolenceThreshold-value');
            if (display) display.textContent = azureViolenceThreshold.value;
        }

        const azureJailbreakThreshold = document.getElementById('azureJailbreakThreshold');
        if (azureJailbreakThreshold) {
            azureJailbreakThreshold.value = config.AZURE_CONTENT_SAFETY_JAILBREAK_THRESHOLD || '0.5';
            const display = document.getElementById('azureJailbreakThreshold-value');
            if (display) display.textContent = azureJailbreakThreshold.value;
        }

        // Update settings visibility
        this.updateLLMGuardSettingsVisibility();
        this.updateAzureSettingsVisibility();
    }

    /**
     * Handle keyboard shortcuts
     */
    handleKeyboardShortcuts(e) {
        // Ctrl/Cmd + S to save
        if ((e.ctrlKey || e.metaKey) && e.key === 's') {
            e.preventDefault();
            // Check which form is currently visible/active
            const globalForm = document.getElementById('configForm');
            const endpointForm = document.getElementById('endpointConfigForm');
            const endpointFormSection = document.getElementById('configFormSection');
            
            // If endpoint form is visible, save endpoint config, otherwise save global config
            if (endpointFormSection && endpointFormSection.style.display !== 'none' && endpointForm) {
                endpointForm.dispatchEvent(new Event('submit', { bubbles: true }));
            } else if (globalForm) {
                globalForm.dispatchEvent(new Event('submit', { bubbles: true }));
            }
        }
        
        // Ctrl/Cmd + R to reload configuration
        if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
            e.preventDefault();
            this.loadConfiguration();
        }
    }

    /**
     * Handle before unload to warn about unsaved changes
     */
    handleBeforeUnload(e) {
        if (this.isDirty) {
            e.preventDefault();
            e.returnValue = 'You have unsaved changes. Are you sure you want to leave?';
            return e.returnValue;
        }
    }

    /**
     * Update status message
     */
    updateStatus(message, type = 'info') {
        const statusElement = document.getElementById('status');
        if (!statusElement) {
            console.log('DEBUG: Status element not found!');
            // Try to create the element if it doesn't exist
            const formActions = document.querySelector('.form-actions');
            if (formActions) {
                const newStatusElement = document.createElement('div');
                newStatusElement.id = 'status';
                newStatusElement.className = 'status';
                formActions.insertBefore(newStatusElement, formActions.firstChild);
                this.updateStatus(message, type); // Recursively call with new element
                return;
            }
            return;
        }

        statusElement.textContent = message;
        statusElement.className = `status ${type}`;
        statusElement.setAttribute('role', type === 'error' ? 'alert' : 'status');
        statusElement.style.display = 'flex'; // Force visibility
        statusElement.style.minHeight = '2.5rem'; // Ensure it has height

        // Auto-clear success messages
        if (type === 'success') {
            setTimeout(() => {
                statusElement.textContent = '';
                statusElement.className = 'status';
            }, 3000);
        }
    }

    /**
     * Reload application configuration in the backend
     */
    async reloadApplicationConfiguration() {
        try {
            this.updateStatus('Reloading application configuration...', 'info');
            
            const response = await fetch('/api/reload-config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP ${response.status}: ${errorText}`);
            }
    
            const result = await response.json();

            if (result.status === 'success') {
                this.updateStatus('Configuration reloaded successfully! Changes are now active.', 'success');
                announceToScreenReader('Application configuration reloaded successfully');
            } else if (result.status === 'already_in_progress') {
                this.updateStatus('Configuration reload already in progress...', 'info');
                // Poll for completion
                this.pollReloadStatus();
            } else {
                throw new Error(result.message || 'Unknown error during reload');
            }

        } catch (error) {
            console.error('Error reloading configuration:', error);
            this.updateStatus(`Failed to reload configuration: ${error.message}`, 'error');
        }
    }

    /**
     * Poll for reload status completion
     */
    async pollReloadStatus() {
        let attempts = 0;
        const maxAttempts = 20; // Max 10 seconds (500ms * 20)
        
        const pollInterval = setInterval(async () => {
            attempts++;

            try {
                const response = await fetch('/api/reload-config/status');
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }

                const status = await response.json();

                if (!status.in_progress) {
                    clearInterval(pollInterval);

                    if (status.last_status === 'success') {
                        this.updateStatus('Configuration reloaded successfully! Changes are now active.', 'success');
                        announceToScreenReader('Application configuration reloaded successfully');
                    } else if (status.last_status.startsWith('error:')) {
                        this.updateStatus(`Configuration reload failed: ${status.last_status}`, 'error');
                    } else {
                        this.updateStatus('Configuration reload completed.', 'success');
                    }
                    return;
                }

                // Update progress message
                this.updateStatus('Configuration reload in progress...', 'info');
                
            } catch (error) {
                console.error('Error polling reload status:', error);
            }

            if (attempts >= maxAttempts) {
                clearInterval(pollInterval);
                this.updateStatus('Configuration reload is taking longer than expected. Please check the server logs.', 'warning');
            }
        }, 500);
    }

    /**
     * Set up configuration management for the Forwarding tab
     */
    setupConfigurationManagement() {
        console.log('Setting up configuration management...');
        
        // Load configurations when the forwarding tab is accessed
        this.loadConfigurations();
        
        // Set up event listeners
        this.setupConfigurationEventListeners();
    }

    /**
     * Set up event listeners for configuration management
     */
    setupConfigurationEventListeners() {
        // Create new configuration
        const createBtn = document.getElementById('createConfigBtn');
        if (createBtn) {
            createBtn.addEventListener('click', () => this.showConfigForm());
        }

        // Refresh configurations
        const refreshBtn = document.getElementById('refreshConfigsBtn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.loadConfigurations());
        }

        // Cancel configuration form
        const cancelBtn = document.getElementById('cancelConfigBtn');
        if (cancelBtn) {
            cancelBtn.addEventListener('click', () => this.hideConfigForm());
        }

        // Test configuration
        const testBtn = document.getElementById('testConfigBtn');
        if (testBtn) {
            testBtn.addEventListener('click', () => this.testConfiguration());
        }

        // Save configuration
        const configForm = document.getElementById('endpointConfigForm');
        if (configForm) {
            configForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.saveConfiguration();
            });
        }

        // Also add a direct click handler to the endpoint save button
        const saveEndpointBtn = document.getElementById('saveEndpointConfigBtn');
        if (saveEndpointBtn) {
            console.log('Setting up endpoint save button click handler');
            saveEndpointBtn.addEventListener('click', (e) => {
                console.log('Endpoint save button clicked directly');
                
                // Prevent any default behavior
                e.preventDefault();
                
                // Call the save configuration method directly
                this.saveConfiguration();
            });
        } else {
            console.log('ERROR: Endpoint save button not found!');
        }

        // Add header button
        const addHeaderBtn = document.getElementById('addHeaderBtn');
        if (addHeaderBtn) {
            addHeaderBtn.addEventListener('click', () => this.addHeaderRow());
        }

        // Set up initial header row remove functionality
        this.setupHeaderRowEvents();
    }

    /**
     * Load and display configurations
     */
    async loadConfigurations() {
        const configsList = document.getElementById('configsList');
        if (!configsList) return;

        try {
            configsList.innerHTML = '<div class="loading-message">Loading configurations...</div>';

            const response = await fetch('/api/configurations?active_only=false');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const responseData = await response.json();
            console.log('Loaded response data:', responseData);
            
            // Extract configurations array from response
            const configurations = responseData.configurations || [];
            
            // Ensure configurations is an array
            if (!Array.isArray(configurations)) {
                throw new Error('Invalid response format: configurations should be an array');
            }

            if (configurations.length === 0) {
                configsList.innerHTML = `
                    <div class="empty-state">
                        <h4>No Configurations</h4>
                        <p>Create your first endpoint configuration to get started.</p>
                    </div>
                `;
                return;
            }

            const configsHtml = configurations.map(config => this.renderConfigItem(config)).join('');
            configsList.innerHTML = configsHtml;

            // Set up event listeners for config actions
            this.setupConfigItemEvents();

        } catch (error) {
            console.error('Error loading configurations:', error);
            configsList.innerHTML = `
                <div class="empty-state">
                    <h4>Error Loading Configurations</h4>
                    <p>${error.message}</p>
                    <button type="button" onclick="window.configUI.loadConfigurations()" class="btn btn-secondary">
                        Retry
                    </button>
                </div>
            `;
        }
    }

    /**
     * Render a configuration item
     */
    renderConfigItem(config) {
        const configData = config.config_data;
        const createdAt = new Date(config.created_at).toLocaleDateString();
        
        return `
            <div class="config-item" data-config-id="${config.id}">
                <div class="config-item-header">
                    <div>
                        <h4 class="config-item-name">${config.name}</h4>
                        <span class="config-status ${config.is_active ? 'active' : 'inactive'}">
                            ${config.is_active ? 'Active' : 'Inactive'}
                        </span>
                    </div>
                    <div class="config-item-actions">
                        <button type="button" class="btn btn-secondary btn-small test-config-btn" 
                                data-config-name="${config.name}" title="Test configuration">
                            🧪 Test
                        </button>
                        <button type="button" class="btn btn-secondary btn-small toggle-config-btn" 
                                data-config-id="${config.id}" data-config-name="${config.name}" 
                                data-is-active="${config.is_active}" 
                                aria-label="${config.is_active ? 'Deactivate' : 'Activate'} configuration ${config.name}"
                                title="${config.is_active ? 'Deactivate' : 'Activate'} configuration">
                            ${config.is_active ? '⏸️ Deactivate' : '▶️ Activate'}
                        </button>
                        <button type="button" class="btn btn-secondary btn-small edit-config-btn" 
                                data-config-name="${config.name}" title="Edit configuration">
                            ✏️ Edit
                        </button>
                        <button type="button" class="btn btn-secondary btn-small delete-config-btn" 
                                data-config-name="${config.name}" title="Delete configuration">
                            🗑️ Delete
                        </button>
                    </div>
                </div>
                <div class="config-item-details">
                    ${config.description ? `<p>${config.description}</p>` : ''}
                    <div class="config-item-url">${configData.url}</div>
                    <div style="margin-top: 0.5rem;">
                        <small>
                            <strong>Method:</strong> ${configData.method} | 
                            <strong>Timeout:</strong> ${configData.timeout}s | 
                            <strong>Headers:</strong> ${Object.keys(configData.headers || {}).length} | 
                            <strong>Created:</strong> ${createdAt}
                        </small>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Set up event listeners for configuration item actions
     */
    setupConfigItemEvents() {
        // Test configuration buttons
        document.querySelectorAll('.test-config-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const configName = e.target.dataset.configName;
                this.testConfigurationByName(configName);
            });
        });

        // Toggle configuration active status buttons
        document.querySelectorAll('.toggle-config-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const configId = e.target.dataset.configId;
                const configName = e.target.dataset.configName;
                const isActive = e.target.dataset.isActive === 'true';
                this.toggleConfigurationStatus(configId, configName, isActive);
            });
        });

        // Edit configuration buttons
        document.querySelectorAll('.edit-config-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const configName = e.target.dataset.configName;
                this.editConfiguration(configName);
            });
        });

        // Delete configuration buttons
        document.querySelectorAll('.delete-config-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const configName = e.target.dataset.configName;
                this.deleteConfiguration(configName);
            });
        });
    }

    /**
     * Show the configuration form
     */
    showConfigForm(editData = null) {
        const formSection = document.getElementById('configFormSection');
        const formTitle = document.getElementById('configFormTitle');
        
        if (!formSection || !formTitle) return;

        if (editData) {
            formTitle.textContent = 'Edit Configuration';
            this.populateConfigForm(editData);
        } else {
            formTitle.textContent = 'Create New Configuration';
            this.clearConfigForm();
        }

        formSection.style.display = 'block';
        formSection.scrollIntoView({ behavior: 'smooth' });
        
        // Focus on the first input
        const firstInput = document.getElementById('configName');
        if (firstInput) {
            firstInput.focus();
        }
    }

    /**
     * Hide the configuration form
     */
    hideConfigForm() {
        const formSection = document.getElementById('configFormSection');
        if (formSection) {
            formSection.style.display = 'none';
        }
        this.clearConfigForm();
    }

    /**
     * Clear the configuration form
     */
    clearConfigForm() {
        const form = document.getElementById('endpointConfigForm');
        if (form) {
            form.reset();
            
            // Reset headers to just one empty row
            const headersContainer = document.getElementById('headersContainer');
            if (headersContainer) {
                headersContainer.innerHTML = `
                    <div class="header-row" data-index="0">
                        <input type="text" placeholder="Header name (e.g., Authorization)" class="header-name">
                        <input type="text" placeholder="Header value (e.g., Bearer \${API_KEY})" class="header-value">
                        <button type="button" class="btn btn-icon remove-header" title="Remove header">❌</button>
                    </div>
                `;
                this.setupHeaderRowEvents();
            }
        }
    }

    /**
     * Populate the configuration form with existing data
     */
    populateConfigForm(config) {
        const configData = config.config_data;
        
        document.getElementById('configName').value = config.name;
        document.getElementById('configDescription').value = config.description || '';
        document.getElementById('configUrl').value = configData.url;
        document.getElementById('configMethod').value = configData.method;
        document.getElementById('configTimeout').value = configData.timeout;
        document.getElementById('configForwardingEnabled').checked = configData.forwarding_enabled || false;
        document.getElementById('configIncludeScanResults').checked = configData.include_scan_results;
        document.getElementById('configForwardOnUnsafe').checked = configData.forward_on_unsafe;
        document.getElementById('configVerifySSL').checked = configData.verify_ssl;

        // Populate headers
        this.populateHeaders(configData.headers || {});
    }

    /**
     * Populate headers in the form
     */
    populateHeaders(headers) {
        const headersContainer = document.getElementById('headersContainer');
        if (!headersContainer) return;

        const headerEntries = Object.entries(headers);
        
        if (headerEntries.length === 0) {
            // Keep the default empty row
            return;
        }

        const headersHtml = headerEntries.map(([ name, value ], index) => `
            <div class="header-row" data-index="${index}">
                <input type="text" placeholder="Header name" class="header-name" value="${name}">
                <input type="text" placeholder="Header value" class="header-value" value="${value}">
                <button type="button" class="btn btn-icon remove-header" title="Remove header">❌</button>
            </div>
        `).join('');

        headersContainer.innerHTML = headersHtml;
        this.setupHeaderRowEvents();
    }

    /**
     * Add a new header row
     */
    addHeaderRow() {
        const headersContainer = document.getElementById('headersContainer');
        if (!headersContainer) return;

        const existingRows = headersContainer.querySelectorAll('.header-row');
        const newIndex = existingRows.length;

        const newRow = document.createElement('div');
        newRow.className = 'header-row';
        newRow.dataset.index = newIndex;
        newRow.innerHTML = `
            <input type="text" placeholder="Header name" class="header-name">
            <input type="text" placeholder="Header value" class="header-value">
            <button type="button" class="btn btn-icon remove-header" title="Remove header">❌</button>
        `;

        headersContainer.appendChild(newRow);
        this.setupHeaderRowEvents();

        // Focus on the new header name input
        const nameInput = newRow.querySelector('.header-name');
        if (nameInput) {
            nameInput.focus();
        }
    }

    /**
     * Set up event listeners for header rows
     */
    setupHeaderRowEvents() {
        document.querySelectorAll('.remove-header').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const headerRow = e.target.closest('.header-row');
                if (headerRow) {
                    const headersContainer = document.getElementById('headersContainer');
                    const remainingRows = headersContainer.querySelectorAll('.header-row');
                    
                    // Keep at least one row
                    if (remainingRows.length > 1) {
                        headerRow.remove();
                    } else {
                        // Clear the last row instead of removing it
                        headerRow.querySelector('.header-name').value = '';
                        headerRow.querySelector('.header-value').value = '';
                    }
                }
            });
        });
    }

    /**
     * Save configuration
     */
    async saveConfiguration() {
        const formData = this.getConfigFormData();
        if (!formData) return;

        const saveBtn = document.getElementById('saveEndpointConfigBtn');
        const originalText = saveBtn.textContent;
        
        try {
            saveBtn.disabled = true;
            saveBtn.textContent = 'Saving...';

            const isEdit = document.getElementById('configFormTitle').textContent.includes('Edit');
            const url = isEdit 
                ? `/api/configurations/endpoints/${formData.name}`
                : '/api/configurations/endpoints';
            const method = isEdit ? 'PUT' : 'POST';

            const response = await fetch(url, {
                method: method,
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to save configuration');
            }

            const result = await response.json();
            console.log('Configuration saved:', result);

            this.hideConfigForm();
            this.loadConfigurations();
            this.updateStatus(`Configuration "${formData.name}" ${isEdit ? 'updated' : 'created'} successfully`, 'success');

        } catch (error) {
            console.error('Error saving configuration:', error);
            this.updateStatus(`Error saving configuration: ${error.message}`, 'error');
        } finally {
            saveBtn.disabled = false;
            saveBtn.textContent = originalText;
        }
    }

    /**
     * Get form data for configuration
     */
    getConfigFormData() {
        const name = document.getElementById('configName').value.trim();
        const description = document.getElementById('configDescription').value.trim();
        const url = document.getElementById('configUrl').value.trim();
        const method = document.getElementById('configMethod').value;
        const timeout = parseInt(document.getElementById('configTimeout').value);

        if (!name || !url) {
            this.updateStatus('Name and URL are required', 'error');
            return null;
        }

        // Collect headers
        const headers = {};
        document.querySelectorAll('.header-row').forEach(row => {
            const nameInput = row.querySelector('.header-name');
            const valueInput = row.querySelector('.header-value');
            
            if (nameInput.value.trim() && valueInput.value.trim()) {
                headers[nameInput.value.trim()] = valueInput.value.trim();
            }
        });

        return {
            name,
            description: description || null,
            endpoint_config: {
                url,
                method,
                timeout,
                headers,
                forwarding_enabled: document.getElementById('configForwardingEnabled').checked,
                include_scan_results: document.getElementById('configIncludeScanResults').checked,
                forward_on_unsafe: document.getElementById('configForwardOnUnsafe').checked,
                verify_ssl: document.getElementById('configVerifySSL').checked,
                retry_attempts: 3,
                retry_delay: 1.0
            }
        };
    }

    /**
     * Test configuration
     */
    async testConfiguration() {
        const formData = this.getConfigFormData();
        if (!formData) return;

        await this.testConfigurationData(formData.endpoint_config);
    }

    /**
     * Test configuration by name
     */
    async testConfigurationByName(configName) {
        try {
            const response = await fetch(`/api/configurations/endpoints/${configName}`);
            if (!response.ok) {
                throw new Error(`Failed to load configuration: ${response.statusText}`);
            }

            const config = await response.json();
            await this.testConfigurationData(config);

        } catch (error) {
            console.error('Error loading configuration for test:', error);
            this.showTestResult(false, `Error loading configuration: ${error.message}`, null);
        }
    }

    /**
     * Test configuration data
     */
    async testConfigurationData(configData) {
        const testBtn = document.getElementById('testConfigBtn');
        const resultDiv = document.getElementById('configTestResult');
        
        try {
            if (testBtn) {
                testBtn.disabled = true;
                testBtn.textContent = 'Testing...';
            }
            
            if (resultDiv) {
                resultDiv.style.display = 'none';
            }

            const response = await fetch('/api/test-endpoint', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    endpoint: configData.url,
                    method: configData.method,
                    headers: configData.headers,
                    timeout: configData.timeout
                })
            });

            const result = await response.json();

            if (response.ok) {
                this.showTestResult(result.success, result.error || 'Test completed successfully', result);
            } else {
                this.showTestResult(false, result.detail || 'Test failed', null);
            }

        } catch (error) {
            console.error('Error testing configuration:', error);
            this.showTestResult(false, `Network error: ${error.message}`, null);
        } finally {
            if (testBtn) {
                testBtn.disabled = false;
                testBtn.textContent = 'Test Configuration';
            }
        }
    }

    /**
     * Show test result
     */
    showTestResult(success, message, details) {
        const resultDiv = document.getElementById('configTestResult');
        if (!resultDiv) return;

        resultDiv.className = `test-result ${success ? 'success' : 'error'}`;
        
        let html = `<h4>${success ? '✓ Test Successful' : '✗ Test Failed'}</h4>`;
        html += `<p>${message}</p>`;
        
        if (details) {
            if (details.status_code) {
                html += `<p><strong>Status Code:</strong> ${details.status_code}</p>`;
            }
            
            if (details.response_data && typeof details.response_data === 'object') {
                html += `<p><strong>Response:</strong></p>`;
                html += `<pre>${JSON.stringify(details.response_data, null, 2)}</pre>`;
            }
        }
        
        resultDiv.innerHTML = html;
        resultDiv.style.display = 'block';
        resultDiv.scrollIntoView({ behavior: 'smooth' });
    }

    /**
     * Edit configuration
     */
    async editConfiguration(configName) {
        try {
            const response = await fetch(`/api/configurations/endpoints/${configName}`);
            if (!response.ok) {
                throw new Error(`Failed to load configuration: ${response.statusText}`);
            }

            const configResponse = await response.json();
            
            // Get the full configuration record
            const configsResponse = await fetch('/api/configurations?active_only=false');
            const allConfigsData = await configsResponse.json();
            const allConfigs = allConfigsData.configurations || [];
            const config = allConfigs.find(c => c.name === configName);

            if (config) {
                this.showConfigForm(config);
            } else {
                throw new Error('Configuration not found');
            }

        } catch (error) {
            console.error('Error loading configuration for edit:', error);
            this.updateStatus(`Error loading configuration: ${error.message}`, 'error');
        }
    }

    /**
     * Delete configuration
     */
    async deleteConfiguration(configName) {
        if (!confirm(`Are you sure you want to delete the configuration "${configName}"?`)) {
            return;
        }

        try {
            const response = await fetch(`/api/configurations/endpoints/${configName}`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to delete configuration');
            }

            this.loadConfigurations();
            this.updateStatus(`Configuration "${configName}" deleted successfully`, 'success');

        } catch (error) {
            console.error('Error deleting configuration:', error);
            this.updateStatus(`Error deleting configuration: ${error.message}`, 'error');
        }
    }

    /**
     * Toggle configuration active status
     */
    async toggleConfigurationStatus(configId, configName, currentlyActive) {
        const newStatus = !currentlyActive;
        const action = newStatus ? 'activate' : 'deactivate';
        
        try {
            const response = await fetch(`/api/configurations/${configId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    is_active: newStatus
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || `Failed to ${action} configuration`);
            }

            // Reload configurations to reflect the change
            this.loadConfigurations();
            this.updateStatus(`Configuration "${configName}" ${action}d successfully`, 'success');
            
            // Announce the change to screen readers
            const statusText = newStatus ? 'activated' : 'deactivated';
            announceToScreenReader(`Configuration ${configName} ${statusText}`);

        } catch (error) {
            console.error(`Error ${action}ing configuration:`, error);
            this.updateStatus(`Error ${action}ing configuration: ${error.message}`, 'error');
        }
    }
}

/**
 * Request Monitoring System
 * Handles real-time request tracking and visualization
 */
class RequestMonitor {
    constructor() {
        this.autoRefreshInterval = null;
        this.isAutoRefreshEnabled = false;
        this.refreshIntervalMs = 5000; // 5 seconds default
        this.init();
    }

    /**
     * Initialize the monitoring system
     */
    init() {
        this.setupMonitoringEventHandlers();
        this.loadRequests();
    }

    /**
     * Set up event handlers for monitoring controls
     */
    setupMonitoringEventHandlers() {
        // Refresh button
        const refreshBtn = document.getElementById('refreshRequests');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.loadRequests();
                announceToScreenReader('Requests refreshed');
            });
        }

        // Clear button
        const clearBtn = document.getElementById('clearRequests');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                this.clearRequests();
                announceToScreenReader('Requests cleared');
            });
        }

        // Auto-refresh toggle
        const autoRefreshToggle = document.getElementById('autoRefreshToggle');
        if (autoRefreshToggle) {
            autoRefreshToggle.addEventListener('click', (e) => {
                this.toggleAutoRefresh();
            });
        }

        // Refresh interval selector
        const intervalSelect = document.getElementById('refresh-interval');
        if (intervalSelect) {
            intervalSelect.addEventListener('change', (e) => {
                this.updateRefreshInterval(parseInt(e.target.value));
            });
        }

        // Filter controls
        const statusFilter = document.getElementById('statusFilter');
        if (statusFilter) {
            statusFilter.addEventListener('change', () => {
                this.applyFilters();
            });
        }

        const dateFilter = document.getElementById('dateFilter');
        if (dateFilter) {
            dateFilter.addEventListener('change', () => {
                this.applyFilters();
            });
        }
    }

    /**
     * Load requests from the API
     */
    async loadRequests() {
        try {
            this.updateLoadingState(true);
            
            const response = await fetch('/api/requests');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            this.displayRequests(data.requests);
            
            // Calculate stats from the requests data
            const stats = this.calculateStats(data.requests, data.total);
            this.updateStats(stats);
            
        } catch (error) {
            console.error('Error loading requests:', error);
            this.showError('Failed to load requests: ' + error.message);
        } finally {
            this.updateLoadingState(false);
        }
    }

    /**
     * Clear all requests
     */
    async clearRequests() {
        console.log('Clearing all requests...');

        // Confirm with user before clearing
        if (!confirm('Are you sure you want to clear all request history? This action cannot be undone.')) {
            return;
        }

        try {
            this.updateLoadingState(true);
            
            const response = await fetch('/api/requests/clear', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP ${response.status}: ${errorText}`);
            }
            
            const result = await response.json();
            console.log('Clear requests result:', result);
            
            // Clear the UI immediately
            this.clearRequestsDisplay();
            
            // Refresh the display to ensure sync with backend
            await this.loadRequests();
            
            announceToScreenReader('All requests cleared successfully');
            this.showSuccess('Request history cleared successfully');
            
        } catch (error) {
            console.error('Error clearing requests:', error);
            this.showError('Failed to clear requests: ' + error.message);
        } finally {
            this.updateLoadingState(false);
        }
    }

    /**
     * Show the result of the endpoint test
     */
    showEndpointTestResult(success, message, details) {
        
        const resultDiv = document.getElementById('endpointTestResult');
        if (!resultDiv) {
            console.error('endpointTestResult element not found');
            return;
        }

        resultDiv.className = `test-result ${success ? 'success' : 'error'}`;
        
        let html = `<h4>${success ? '✓ Test Successful' : '✗ Test Failed'}</h4>`;
        html += `<p>${message}</p>`;
        
        if (details) {
            if (details.status_code) {
                html += `<p><strong>Status Code:</strong> ${details.status_code}</p>`;
            }
            
            if (details.response_data && typeof details.response_data === 'object') {
                html += `<p><strong>Response:</strong></p>`;
                html += `<pre>${JSON.stringify(details.response_data, null, 2)}</pre>`;
            }
        }
        
        resultDiv.innerHTML = html;
        resultDiv.style.display = 'block';
        
        // Announce to screen readers
        announceToScreenReader(success ? 'Endpoint test successful' : 'Endpoint test failed');
    }

    /**
     * Display requests in the UI
     */
    displayRequests(requests) {
        const container = document.getElementById('requestsList');
        if (!container) return;

        if (!requests || requests.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <p>No requests found. Make a scan request to see tracking data here.</p>
                </div>
            `;
            return;
        }

        container.innerHTML = requests.map(request => this.createRequestItem(request)).join('');
        this.applyFilters();
    }

    /**
     * Create HTML for a single request item
     */
    createRequestItem(request) {
        const timestamp = new Date(request.timestamp).toLocaleString();
        const duration = request.processing_time_ms ? `${request.processing_time_ms.toFixed(2)}ms` : 'N/A';
        const statusClass = request.status === 'success' ? 'success' : 
                           request.status === 'error' ? 'error' : 'processing';

        return `
            <div class="request-item" data-status="${request.status}" data-timestamp="${request.timestamp}">
                <div class="request-header">
                    <div class="request-id">
                        <strong>Request ID:</strong> 
                        <code>${request.id}</code>
                    </div>
                    <div class="request-status">
                        <span class="status-badge ${statusClass}">${request.status}</span>
                    </div>
                </div>
                
                <div class="request-metadata">
                    <div class="metadata-grid">
                        <div class="metadata-item">
                            <span class="metadata-label">Timestamp:</span>
                            <span class="metadata-value">${timestamp}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Duration:</span>
                            <span class="metadata-value">${duration}</span>
                        </div>
                        <div class="metadata-item">
                            <span class="metadata-label">Message Length:</span>
                            <span class="metadata-value">${request.content_preview?.length || request.content_length || 0} chars</span>
                        </div>
                    </div>
                </div>

                <div class="request-content">
                    <div class="message-section">
                        <h4>Input Message:</h4>
                        <div class="message-content">${this.escapeHtml(request.content_preview || 'N/A')}</div>
                    </div>
                    
                    ${request.response ? `
                        <div class="response-section">
                            <h4>Security Scan Results:</h4>
                            <div class="response-summary">
                                <div class="summary-item">
                                    <span class="summary-label">Overall Safe:</span>
                                    <span class="summary-value ${request.response.is_safe ? 'safe' : 'unsafe'}">
                                        ${request.response.is_safe ? 'Yes' : 'No'}
                                    </span>
                                </div>
                                <div class="summary-item">
                                    <span class="summary-label">Risk Score:</span>
                                    <span class="summary-value">${request.response.score?.toFixed(3) || 'N/A'}</span>
                                </div>
                                <div class="summary-item">
                                    <span class="summary-label">Systems Used:</span>
                                    <span class="summary-value">${request.response.scan_type || 'N/A'}</span>
                                </div>
                            </div>
                            
                            ${request.response.scan_results ? this.createScanResultsSection(request.response.scan_results) : ''}
                            
                            ${request.response.risks_found?.length > 0 ? `
                                <div class="risks-section">
                                    <h5>Security Risks Summary:</h5>
                                    <ul class="risks-list">
                                        ${request.response.risks_found.map(risk => 
                                            `<li class="risk-item">${this.escapeHtml(risk)}</li>`
                                        ).join('')}
                                    </ul>
                                </div>
                            ` : ''}
                        </div>
                    ` : ''}
                    
                    ${request.error ? `
                        <div class="error-section">
                            <h4>Error Details:</h4>
                            <div class="error-content">${this.escapeHtml(request.error)}</div>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    }

    /**
     * Create detailed scan results section for each security system
     */
    createScanResultsSection(scanResults) {
        let html = '<div class="scan-systems-section"><h5>Detailed Scan Results by System:</h5>';
        
        // LlamaFirewall Results
        if (scanResults.llamafirewall) {
            const llama = scanResults.llamafirewall;
            html += `
                <div class="scan-system">
                    <h6>🦙 LlamaFirewall</h6>
                    <div class="system-details">
                        <div class="detail-item">
                            <span class="detail-label">Decision:</span>
                            <span class="detail-value ${llama.is_safe ? 'safe' : 'unsafe'}">${llama.decision}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Risk Score:</span>
                            <span class="detail-value">${llama.score?.toFixed(3) || 'N/A'}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Reason:</span>
                            <span class="detail-value">${this.escapeHtml(llama.reason || 'N/A')}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Safe:</span>
                            <span class="detail-value ${llama.is_safe ? 'safe' : 'unsafe'}">${llama.is_safe ? 'Yes' : 'No'}</span>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // LLM Guard Results
        if (scanResults.llmguard) {
            const guard = scanResults.llmguard;
            if (guard.enabled) {
                html += `
                    <div class="scan-system">
                        <h6>🛡️ LLM Guard</h6>
                        <div class="system-details">
                            <div class="detail-item">
                                <span class="detail-label">Overall Safe:</span>
                                <span class="detail-value ${guard.is_safe ? 'safe' : 'unsafe'}">${guard.is_safe ? 'Yes' : 'No'}</span>
                            </div>
                `;
                
                if (guard.validation_results && Object.keys(guard.validation_results).length > 0) {
                    html += '<div class="detail-item"><span class="detail-label">Scanner Results:</span><div class="scanner-results">';
                    for (const [scanner, passed] of Object.entries(guard.validation_results)) {
                        const score = guard.risk_scores && guard.risk_scores[scanner] ? ` (${guard.risk_scores[scanner].toFixed(3)})` : '';
                        html += `<div class="scanner-result ${passed ? 'passed' : 'failed'}">
                            ${scanner}: ${passed ? 'PASS' : 'FAIL'}${score}
                        </div>`;
                    }
                    html += '</div></div>';
                }
                
                if (guard.failed_scanners && guard.failed_scanners.length > 0) {
                    html += `
                        <div class="detail-item">
                            <span class="detail-label">Failed Scanners:</span>
                            <span class="detail-value failed">${guard.failed_scanners.join(', ')}</span>
                        </div>
                    `;
                }
                
                html += '</div></div>';
            } else {
                html += `
                    <div class="scan-system">
                        <h6>🛡️ LLM Guard</h6>
                        <div class="system-details">
                            <div class="detail-item">
                                <span class="detail-label">Status:</span>
                                <span class="detail-value disabled">Disabled</span>
                            </div>
                        </div>
                    </div>
                `;
            }
        }
        
        // OpenAI Moderation Results
        if (scanResults.openai_moderation) {
            const mod = scanResults.openai_moderation;
            if (mod.enabled) {
                html += `
                    <div class="scan-system">
                        <h6>🤖 OpenAI Moderation</h6>
                        <div class="system-details">
                            <div class="detail-item">
                                <span class="detail-label">Safe:</span>
                                <span class="detail-value ${mod.is_safe ? 'safe' : 'unsafe'}">${mod.is_safe ? 'Yes' : 'No'}</span>
                            </div>
                            <div class="detail-item">
                                <span class="detail-label">Model:</span>
                                <span class="detail-value">${mod.model || 'N/A'}</span>
                            </div>
                `;
                
                if (mod.flagged_categories && Object.keys(mod.flagged_categories).length > 0) {
                    html += '<div class="detail-item"><span class="detail-label">Flagged Categories:</span><div class="flagged-categories">';
                    for (const [category, score] of Object.entries(mod.flagged_categories)) {
                        html += `<div class="flagged-category">
                            ${category}: ${score.toFixed(3)}
                        </div>`;
                    }
                    html += '</div></div>';
                }
                
                html += '</div></div>';
            } else {
                html += `
                    <div class="scan-system">
                        <h6>🤖 OpenAI Moderation</h6>
                        <div class="system-details">
                            <div class="detail-item">
                                <span class="detail-label">Status:</span>
                                <span class="detail-value disabled">Disabled</span>
                            </div>
                        </div>
                    </div>
                `;
            }
        }

        // Azure AI Content Safety Results
        if (scanResults.azure) {
            const azure = scanResults.azure;
            if (azure.enabled) {
                html += `
                    <div class="scan-system">
                        <h6>☁️ Azure AI Content Safety</h6>
                        <div class="system-details">
                            <div class="detail-item">
                                <span class="detail-label">Overall Safe:</span>
                                <span class="detail-value ${azure.is_safe ? 'safe' : 'unsafe'}">${azure.is_safe ? 'Yes' : 'No'}</span>
                            </div>
                `;

                if (azure.severity_scores && Object.keys(azure.severity_scores).length > 0) {
                    html += '<div class="detail-item"><span class="detail-label">Content Categories:</span><div class="scanner-results">';
                    for (const [category, severity] of Object.entries(azure.severity_scores)) {
                        html += `<div class="scanner-result">
                            ${category}: ${severity}/4
                        </div>`;
                    }
                    html += '</div></div>';
                }

                if (azure.flagged_categories && azure.flagged_categories.length > 0) {
                    html += '<div class="detail-item"><span class="detail-label">Flagged Categories:</span><div class="flagged-categories">';
                    for (const flagged of azure.flagged_categories) {
                        html += `<div class="flagged-category failed">
                            ${flagged.category}: ${flagged.severity}/${flagged.threshold}
                        </div>`;
                    }
                    html += '</div></div>';
                }

                // Show enabled features
                const features = [];
                if (azure.text_enabled) features.push('Text Analysis');
                if (azure.jailbreak_enabled) features.push('Jailbreak Detection');

                if (features.length > 0) {
                    html += `
                        <div class="detail-item">
                            <span class="detail-label">Features:</span>
                            <span class="detail-value">${features.join(', ')}</span>
                        </div>
                    `;
                }

                html += '</div></div>';
            } else {
                html += `
                    <div class="scan-system">
                        <h6>☁️ Azure AI Content Safety</h6>
                        <div class="system-details">
                            <div class="detail-item">
                                <span class="detail-label">Status:</span>
                                <span class="detail-value disabled">Disabled</span>
                            </div>
                        </div>
                    </div>
                `;
            }
        }
        
        // Forwarding Results - only show if forwarding was attempted
        if (scanResults.forwarding && scanResults.forwarding.enabled) {
            const forward = scanResults.forwarding;
            html += `
                <div class="scan-system">
                    <h6>🔗 Request Forwarding</h6>
                    <div class="system-details">
                        <div class="detail-item">
                            <span class="detail-label">Endpoint:</span>
                            <span class="detail-value">${this.escapeHtml(forward.endpoint || 'N/A')}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Status:</span>
                            <span class="detail-value ${forward.success ? 'safe' : 'unsafe'}">${forward.success ? 'Success' : 'Failed'}</span>
                        </div>
            `;
            
            if (forward.status_code) {
                html += `
                    <div class="detail-item">
                        <span class="detail-label">HTTP Status:</span>
                        <span class="detail-value">${forward.status_code}</span>
                    </div>
                `;
            }
            
            if (forward.error) {
                html += `
                    <div class="detail-item">
                        <span class="detail-label">Error:</span>
                        <span class="detail-value failed">${this.escapeHtml(forward.error)}</span>
                    </div>
                `;
            }
            
            html += '</div></div>';
        }
        
        html += '</div>';
        return html;
    }

    /**
     * Update statistics display
     */
    updateStats(stats) {
        const totalElement = document.getElementById('totalRequests');
        const successRateElement = document.getElementById('successRate');
        const avgTimeElement = document.getElementById('avgResponseTime');

        if (totalElement) totalElement.textContent = stats.total || 0;
        if (successRateElement) {
            const total = stats.total || 0;
            const success = stats.success || 0;
            const rate = total > 0 ? ((success / total) * 100).toFixed(1) : 0;
            successRateElement.textContent = `${rate}%`;
        }
        if (avgTimeElement) {
            avgTimeElement.textContent = stats.avg_response_time ? 
                `${stats.avg_response_time.toFixed(2)}ms` : '0ms';
        }
    }

    /**
     * Calculate statistics from requests data
     */
    calculateStats(requests, total) {
        if (!requests || requests.length === 0) {
            return {
                total: total || 0,
                success: 0,
                avg_response_time: 0
            };
        }

        const successCount = requests.filter(r => r.status === 'success').length;
        const totalResponseTime = requests.reduce((sum, r) => sum + (r.processing_time_ms || 0), 0);
        const avgResponseTime = requests.length > 0 ? totalResponseTime / requests.length : 0;

        return {
            total: total || requests.length,
            success: successCount,
            avg_response_time: avgResponseTime
        };
    }

    /**
     * Apply filters to request display
     */
    applyFilters() {
        const statusFilter = document.getElementById('statusFilter')?.value || 'all';
        const dateFilter = document.getElementById('dateFilter')?.value || 'all';
        const items = document.querySelectorAll('.request-item');

        let visibleCount = 0;

        items.forEach(item => {
            let visible = true;

            // Status filter
            if (statusFilter !== 'all') {
                const itemStatus = item.dataset.status;
                if (itemStatus !== statusFilter) {
                    visible = false;
                }
            }

            // Date filter
            if (dateFilter !== 'all') {
                const itemTimestamp = new Date(item.dataset.timestamp);
                const now = new Date();
                const cutoff = new Date();

                switch (dateFilter) {
                    case '1h':
                        cutoff.setHours(now.getHours() - 1);
                        break;
                    case '24h':
                        cutoff.setDate(now.getDate() - 1);
                        break;
                    case '7d':
                        cutoff.setDate(now.getDate() - 7);
                        break;
                }

                if (itemTimestamp < cutoff) {
                    visible = false;
                }
            }

            item.style.display = visible ? 'block' : 'none';
            if (visible) visibleCount++;
        });

        // Update visible count
        const countElement = document.getElementById('filtered-count');
        if (countElement) {
            countElement.textContent = visibleCount;
        }
    }

    /**
     * Toggle auto-refresh functionality
     */
    toggleAutoRefresh(enabled = null) {
        // If no parameter provided, toggle current state
        if (enabled === null) {
            enabled = !this.isAutoRefreshEnabled;
        }
        
        this.isAutoRefreshEnabled = enabled;
        
        const toggleButton = document.getElementById('autoRefreshToggle');
        
        if (enabled) {
            this.autoRefreshInterval = setInterval(() => {
                this.loadRequests();
            }, this.refreshIntervalMs);
            
            if (toggleButton) {
                toggleButton.innerHTML = '⏸️ Auto-Refresh: ON';
                toggleButton.classList.add('active');
            }
            announceToScreenReader('Auto-refresh enabled');
        } else {
            if (this.autoRefreshInterval) {
                clearInterval(this.autoRefreshInterval);
                this.autoRefreshInterval = null;
            }
            
            if (toggleButton) {
                toggleButton.innerHTML = '▶️ Auto-Refresh: OFF';
                toggleButton.classList.remove('active');
            }
            announceToScreenReader('Auto-refresh disabled');
        }
    }

    /**
     * Update refresh interval
     */
    updateRefreshInterval(intervalMs) {
        this.refreshIntervalMs = intervalMs;
        
        if (this.isAutoRefreshEnabled) {
            // Restart auto-refresh with new interval
            this.toggleAutoRefresh(false);
            this.toggleAutoRefresh(true);
        }
    }

    /**
     * Update loading state
     */
    updateLoadingState(isLoading) {
        const refreshBtn = document.getElementById('refreshRequests');
        const clearBtn = document.getElementById('clearRequests');
        const loadingIndicator = document.getElementById('loading-indicator');

        if (refreshBtn) {
            refreshBtn.disabled = isLoading;
            refreshBtn.setAttribute('aria-busy', isLoading);
        }
        
        if (clearBtn) {
            clearBtn.disabled = isLoading;
        }
        
        if (loadingIndicator) {
            loadingIndicator.style.display = isLoading ? 'block' : 'none';
        }
    }

    /**
     * Show error message
     */
    showError(message) {
        const errorContainer = document.getElementById('monitoring-errors');
        if (!errorContainer) return;

        errorContainer.style.display = 'block';

        const errorElement = document.createElement('div');
        errorElement.className = 'error-message';
        errorElement.setAttribute('role', 'alert');
        errorElement.innerHTML = `
            <span class="error-icon" aria-hidden="true">⚠️</span>
            <span class="error-text">${this.escapeHtml(message)}</span>
            <button class="error-dismiss" aria-label="Dismiss error">×</button>
        `;

        errorContainer.appendChild(errorElement);

        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (errorElement.parentNode) {
                errorElement.parentNode.removeChild(errorElement);
                if (errorContainer.children.length === 0) {
                    errorContainer.style.display = 'none';
                }
            }
        }, 5000);

        // Manual dismiss
        errorElement.querySelector('.error-dismiss').addEventListener('click', () => {
            if (errorElement.parentNode) {
                errorElement.parentNode.removeChild(errorElement);
                if (errorContainer.children.length === 0) {
                    errorContainer.style.display = 'none';
                }
            }
        });
    }

    /**
     * Show success message
     */
    showSuccess(message) {
        const errorContainer = document.getElementById('monitoring-errors');
        if (!errorContainer) return;

        errorContainer.style.display = 'block';

        const successElement = document.createElement('div');
        successElement.className = 'success-message';
        successElement.setAttribute('role', 'status');
        successElement.innerHTML = `
            <span class="success-icon" aria-hidden="true">✅</span>
            <span class="success-text">${this.escapeHtml(message)}</span>
            <button class="success-dismiss" aria-label="Dismiss message">×</button>
        `;

        errorContainer.appendChild(successElement);

        // Auto-dismiss after 3 seconds
        setTimeout(() => {
            if (successElement.parentNode) {
                successElement.parentNode.removeChild(successElement);
                if (errorContainer.children.length === 0) {
                    errorContainer.style.display = 'none';
                }
            }
        }, 3000);

        // Manual dismiss
        successElement.querySelector('.success-dismiss').addEventListener('click', () => {
            if (successElement.parentNode) {
                successElement.parentNode.removeChild(successElement);
                if (errorContainer.children.length === 0) {
                    errorContainer.style.display = 'none';
                }
            }
        });
    }

    /**
     * Clear the requests display immediately
     */
    clearRequestsDisplay() {
        const requestsList = document.getElementById('requestsList');
        if (requestsList) {
            requestsList.innerHTML = '<div class="empty-state">No requests found</div>';
        }

        // Reset stats
        const totalRequests = document.getElementById('totalRequests');
        const successRate = document.getElementById('successRate');
        const avgResponseTime = document.getElementById('avgResponseTime');

        if (totalRequests) totalRequests.textContent = '0';
        if (successRate) successRate.textContent = '0%';
        if (avgResponseTime) avgResponseTime.textContent = '0ms';
    }

    /**
     * Escape HTML to prevent XSS
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Utility functions for enhanced accessibility

/**
 * Announce message to screen readers
 * @param {string} message - Message to announce
 */
function announceToScreenReader(message) {
    const announcement = document.createElement('div');
    announcement.setAttribute('aria-live', 'polite');
    announcement.setAttribute('aria-atomic', 'true');
    announcement.className = 'sr-only';
    announcement.textContent = message;
    
    document.body.appendChild(announcement);
    
    setTimeout(() => {
        document.body.removeChild(announcement);
    }, 1000);
}

/**
 * Enhanced focus management
 */
function manageFocus() {
    // Track focus for keyboard navigation
    let lastFocusedElement = null;
    
    document.addEventListener('focusin', (e) => {
        lastFocusedElement = e.target;
    });
    
    // Return focus when closing modals or overlays
    window.returnFocus = () => {
        if (lastFocusedElement && document.contains(lastFocusedElement)) {
            lastFocusedElement.focus();
        }
    };
}

// Initialize focus management
manageFocus();

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing LLM Firewall UI...');
    
    const configUI = new ConfigUI();
    
    // Make configUI available globally for configuration management
    window.configUI = configUI;
    
    // Initialize monitoring if the monitoring tab is present
    const monitoringPanel = document.getElementById('panel-monitoring');
    if (monitoringPanel) {
        console.log('Initializing request monitoring...');
        new RequestMonitor();
    }
    
    console.log('UI initialization complete.');
});
