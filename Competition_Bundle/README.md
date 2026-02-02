# Competition Bundle - Grain Variety Classification Challenge

## Overview
***

This directory contains the complete **Competition Bundle** for the Grain Variety Classification Challenge. This bundle can be compiled and uploaded to Codabench to create the competition.

## What is a Competition Bundle?
***

A Competition Bundle is a collection of files and folders that define a complete competition on Codabench. It includes:
- **competition.yaml**: Competition configuration (title, description, phases, tasks, etc.)
- **logo.png**: Competition logo
- **pages/**: Markdown pages for the competition website (Overview, Data, Starting Kit, etc.)
- **ingestion_program/**: Code that runs participant submissions
- **scoring_program/**: Code that evaluates submissions and computes scores
- **input_data/**: Training and test data (or links to data)
- **reference_data/**: Ground truth labels for evaluation
- **sample_code_submission/**: Example submission for participants

## Bundle Structure
***

```
Competition_Bundle/
├── competition.yaml          # Main configuration file
├── logo.png                  # Competition logo
├── README.md                 # This file
├── create_bundle.py          # Script to create bundle zip
├── create_submission_zip.py  # Script to create participant submission zip
├── pages/                    # Competition website pages
│   ├── overview.md          # Competition overview
│   ├── evaluation.md        # Evaluation metrics description
│   ├── data.md              # Data description
│   ├── starting_kit.md      # Starting kit instructions
│   └── terms.md             # Terms and conditions
├── ingestion_program/        # Code that runs participant models
│   ├── ingestion.py        # Main ingestion logic
│   ├── run_ingestion.py    # Entry point
│   └── README.md           # Ingestion program documentation
├── scoring_program/         # Code that evaluates submissions
│   ├── score.py            # Scoring logic
│   ├── run_scoring.py      # Entry point
│   └── README.md           # Scoring program documentation
├── input_data/              # Training and test data
│   ├── train/              # Training images
│   ├── test/               # Test images
│   └── README.md           # Data documentation
├── reference_data/          # Ground truth labels
│   ├── train_labels.json   # Training labels
│   ├── test_labels.json    # Test labels (for scoring only)
│   └── README.md           # Reference data documentation
└── sample_code_submission/  # Example submission
    ├── model.py            # Baseline model
    ├── requirements.txt    # Dependencies
    └── ...                 # Other model variants
```

## How to Compile the Bundle
***

### Option 1: Using the Provided Script (Recommended)

The easiest way to create the bundle zip is using the provided Python script:

```bash
cd Competition_Bundle
python3 create_bundle.py
```

This will:
1. Create a zip file named `grain_classification_bundle.zip`
2. Include all necessary files and folders
3. Ensure `competition.yaml` is at the root of the zip (required by Codabench)
4. Exclude unnecessary files (like `__pycache__`, `.DS_Store`, etc.)

**Output**: `grain_classification_bundle.zip` in the `Competition_Bundle/` directory

### Option 2: Manual Creation

If you prefer to create the zip manually:

```bash
cd Competition_Bundle

# Create zip with all contents (NOT the parent directory!)
zip -r grain_classification_bundle.zip . \
  -x "*.pyc" \
  -x "__pycache__/*" \
  -x "*.DS_Store" \
  -x ".git/*" \
  -x "*.zip"
```

**Important**: 
- Zip the **contents** of `Competition_Bundle/`, not the folder itself
- `competition.yaml` must be at the **root** of the zip file
- Do NOT include parent directories in the zip

### Option 3: Using the Shell Script

A shell script is also provided:

```bash
cd Competition_Bundle
bash create_bundle.sh
```

## How to Upload to Codabench
***

1. **Login to Codabench**: Go to [https://www.codabench.org/](https://www.codabench.org/)
2. **Go to Competitions**: Click on "Competitions" in the navigation
3. **Create Competition**: Click "Create Competition" or "Upload Bundle"
4. **Upload Zip**: Select `grain_classification_bundle.zip`
5. **Wait for Processing**: Codabench will unpack and validate the bundle
6. **Review**: Check that all pages, tasks, and phases are correctly configured
7. **Test**: Upload a sample submission to verify everything works

## Verification Checklist
***

Before uploading, verify:

- [ ] `competition.yaml` is at the root of the zip (not in a subdirectory)
- [ ] All required pages exist in `pages/` folder
- [ ] Logo file (`logo.png`) is included
- [ ] Ingestion program is complete and tested
- [ ] Scoring program is complete and tested
- [ ] Sample submission works (test locally if possible)
- [ ] Data is included or properly linked (if data is too large, use Resources)
- [ ] All file paths in `competition.yaml` are correct

## Testing the Bundle Locally
***

### Test Ingestion Program

```bash
cd Competition_Bundle/ingestion_program
python3 run_ingestion.py --local
```

### Test Scoring Program

```bash
cd Competition_Bundle/scoring_program
python3 run_scoring.py --local
```

### Test Sample Submission

1. Create a submission zip:
   ```bash
   cd Competition_Bundle
   python3 create_submission_zip.py
   ```

2. Test it locally (if you have a local test setup)

## Common Issues and Solutions
***

### Issue: "competition.yaml is missing from zip"
**Solution**: The zip was created incorrectly. Use `create_bundle.py` script or ensure you zip the contents, not the parent directory.

### Issue: "YAML syntax error"
**Solution**: Check `competition.yaml` for syntax errors. Common issues:
- Missing quotes around strings with colons
- Incorrect indentation
- Invalid date formats

### Issue: "Phases dates conflict"
**Solution**: Ensure phase end dates don't overlap with next phase start dates.

### Issue: "Page not found"
**Solution**: Verify all page files exist in `pages/` folder and paths in `competition.yaml` are correct.

### Issue: "Submission fails on Codabench"
**Solution**: 
- Check ingestion program logs
- Verify sample submission format
- Test locally if possible
- Review error messages in Codabench logs

## Data Management
***

### Large Datasets

If your dataset is too large to include in the bundle:

1. **Upload Data Separately**:
   - Go to Codabench Resources page
   - Upload your data files
   - Make them public
   - Link them in phase settings

2. **Use Data Links**:
   - Update `competition.yaml` to reference external data
   - Or provide download instructions in Data page

### Private Data

If data is private or sensitive:
- Don't include it in the bundle
- Use Codabench Resources with access controls
- Contact Codabench administrators for help

## Updating the Bundle
***

After making changes:

1. **Update Files**: Modify the necessary files (pages, code, etc.)
2. **Recreate Zip**: Run `create_bundle.py` again
3. **Re-upload**: Upload the new zip to Codabench
4. **Test**: Verify changes work correctly

**Note**: Codabench may require you to create a new competition or update an existing one.

## Scripts Provided
***

### `create_bundle.py`
Creates the competition bundle zip file.

**Usage**:
```bash
python3 create_bundle.py
```

**Options**: See script for available options (output name, exclusions, etc.)

### `create_submission_zip.py`
Creates a participant submission zip file.

**Usage**:
```bash
python3 create_submission_zip.py              # Uses baseline model
python3 create_submission_zip.py --baseline   # Explicit baseline
python3 create_submission_zip.py --simple    # Simple model
```

### `create_mini_dataset.py`
Creates a smaller dataset for quick testing (temporary script).

**Usage**:
```bash
python3 create_mini_dataset.py        # Create mini dataset
python3 create_mini_dataset.py --restore  # Restore full dataset
```

## Additional Resources
***

- **Codabench Documentation**: [https://www.codabench.org/docs/](https://www.codabench.org/docs/)
- **Example Competitions**: 
  - [Competition 1145](https://www.codabench.org/competitions/1145/)
  - [Competition 2044](https://www.codabench.org/competitions/2044/)
- **GitHub Repository**: See main repository README for more information

## Getting Help
***

If you encounter issues:

1. **Check Logs**: Review Codabench logs for error messages
2. **Test Locally**: Try to reproduce issues locally
3. **Review Documentation**: Check this README and other documentation
4. **Contact Organizers**: Reach out via Codabench or email
5. **GitHub Issues**: Report bugs or ask questions on GitHub

## Next Steps
***

1. ✅ **Review Bundle Structure**: Ensure all files are in place
2. ✅ **Test Locally**: Verify ingestion and scoring programs work
3. ✅ **Create Bundle Zip**: Use `create_bundle.py`
4. ✅ **Upload to Codabench**: Follow upload instructions
5. ✅ **Test Submission**: Upload sample submission to verify everything works
6. ✅ **Publish Competition**: Make it available to participants

---

**Good luck with your competition! 🚀**
