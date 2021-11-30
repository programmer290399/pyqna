read -p "Enter Release version: " version
git commit --allow-empty -m "REL: $version"
git tag -as v$version
echo "Version:"
python -c "import pyqna;print(pyqna.__version__)"  
read -p "Continue? (Y/N): " confirm && [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]] || exit 1
git push origin main
git push origin v$version
git clean -dfn
read -p "Continue? (Y/N): " confirm && [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]] || exit 1
git clean -dfx
python setup.py sdist
python setup.py bdist_wheel
twine upload --skip-existing dist/*
