PY = python3.6 -O -m compileall -b -q -f
SRC = 
TARGETS = 
TEMP = 
all: clean $(TARGETS)

$(TARGETS):
	@echo "Compiling ..."
	@cp -r $(SRC) $(TARGETS)
	-$(PY) $(TARGETS)
	@find $(TARGETS) -name '*.py' -delete
	@find $(TARGETS) -name "__pycache__" |xargs rm -rf

clean:
	@echo "Clean ..." 
	@find $(SRC) -name "__pycache__" | xargs rm -rf
	@find $(SRC) -name '*.pyc' -delete
	@rm -rf $(TARGETS) $(TEMP)
